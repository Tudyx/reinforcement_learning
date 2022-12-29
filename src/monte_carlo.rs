//! Monte Carlo agent.
//! There is no discouting factor for the Easy21 assignement.
//!
//! A lot of onlin implementation seems to have errors.
//! This one seems to be good https://github.com/hereismari/easy21/blob/master/easy21.ipynb

// FIXME: in the plot we are not enought close to one.
// TODO: benchmark le passage de Copy a CLone
// TODO: benchmark si jamais je passe des reference pour le lookup.

use itertools::izip;
use ndarray::{Array, Array2};
use plotters::prelude::*;
use rand::prelude::*;
use rust_gym::easy_21::{Action, Easy21, Observation};
use rust_gym::Environment;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
/// Retex:
/// Ord n'est pas implémenter pour f64!

/// Define how much we explore.
const N_0: f64 = 100.;
/// Print status every x episode.
const EPISODE_PRINT: u64 = 10_000;

/// Represent a trajectory of through an episode.
// Use a struct of array instead of an array of struct for efficiency.
// TODO: benchmark to be sure its more efficient.
#[derive(Debug)]
struct Trajectory {
    states: Vec<Observation>,
    actions: Vec<Action>,
    rewards: Vec<f64>,
}

impl Trajectory {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, step: (Observation, Action, f64)) {
        self.states.push(step.0);
        self.actions.push(step.1);
        self.rewards.push(step.2);
    }

    /// Return the episode *return* A.K.A. the cumulated reward.
    /// No discount factor is applied.
    fn cumulated_reward(&self) -> f64 {
        self.rewards.iter().sum()
    }

    /// Iter on the trajectory in the reverse order.
    fn iter_rev(&self) -> impl Iterator<Item = (&Observation, &Action, &f64)> {
        izip!(&self.states, &self.actions, &self.rewards).rev()
    }

    /// Iterate over states and action
    fn iter(&self) -> impl Iterator<Item = (&Observation, &Action)> {
        self.states.iter().zip(self.actions.iter())
    }
}

/// An agent using Monte Carlo control to find the best policy.
struct MonteCarloAgent {
    /// Our action-value function (Q value) that we will try to improve towards Q*(s, a).
    action_value: HashMap<(Observation, Action), f64>,
    /// The N(s) function. Give number of time we have visited a state.
    visited_states: HashMap<Observation, i32>,
    // The N(s,a) function. Give the number of time we have visited a couple action state.
    visited_state_action: HashMap<(Observation, Action), u64>,
    /// The underlying environment.
    env: Easy21,
}

impl MonteCarloAgent {
    fn new(env: Easy21) -> Self {
        Self {
            action_value: HashMap::with_capacity(10 * 21 * 2),
            visited_states: HashMap::with_capacity(10 * 21),
            visited_state_action: HashMap::with_capacity(10 * 21 * 2),
            env,
        }
    }

    /// Improve the action-value function approximation towards Q*(s, a) from the given trajectory.
    fn update_action_value(&mut self, trajectory: Trajectory) {
        let mut cumulated_reward = 0.;

        // We iterated in the reverse order!
        for (state, action, reward) in trajectory.iter_rev() {
            // The cumultated reward we have from that state action pair.
            cumulated_reward += reward;

            self.visited_states
                .entry(state.clone())
                .and_modify(|count| *count += 1)
                .or_insert(1);

            self.visited_state_action
                .entry((state.clone(), action.clone()))
                .and_modify(|count| *count += 1)
                .or_insert(1);

            let alpha =
                1. / self.visited_state_action[&(state.to_owned(), action.to_owned())] as f64;

            // We adjust the Q value towards the reality (observed) minus what we estimated.
            // This term is usually descrived as the error term.
            self.action_value
                .entry((state.clone(), action.clone()))
                .and_modify(|value| *value += alpha * (cumulated_reward - *value))
                .or_insert(0.);
        }
    }

    fn choose_random_action() -> Action {
        if random() {
            Action::Stick
        } else {
            Action::Hit
        }
    }

    /// We pick the action with the max action-value.
    fn choose_greedy_action(&self, observation: &Observation) -> Action {
        let stick_value = self
            .action_value
            .get(&(observation.clone(), Action::Stick))
            .unwrap_or(&0.);

        let hit_value = self
            .action_value
            .get(&(observation.clone(), Action::Hit))
            .unwrap_or(&0.);

        if stick_value > hit_value {
            Action::Stick
        } else {
            Action::Hit
        }
    }

    /// We explore the state space with a probability of epsilon. Otherwise we take a greddy action (the best we can).
    fn epsilon_greedy_policy(&self, observation: &Observation) -> Action {
        // The less we have seen a state, the more we explore.
        let epsilon = N_0 / (N_0 + *self.visited_states.get(observation).unwrap_or(&0) as f64);

        if thread_rng().gen::<f64>() <= epsilon {
            // Exploration.
            Self::choose_random_action()
        } else {
            // Exploitation.
            self.choose_greedy_action(&observation)
        }
    }

    /// We compute the estimated state value function from the estimated action value function.
    fn compute_state_value_function(&self) -> Array2<f64> {
        let mut state_value = Array::zeros((21, 10));

        for (observation, _) in &self.visited_states {
            let stick_value = self
                .action_value
                .get(&(observation.clone(), Action::Stick))
                .unwrap_or(&0.);

            let hit_value = self
                .action_value
                .get(&(observation.clone(), Action::Hit))
                .unwrap_or(&0.);

            let max_q_value = if stick_value > hit_value {
                stick_value
            } else {
                hit_value
            };
            state_value[[
                observation.player_sum as usize - 1,
                observation.bank_sum as usize - 1,
            ]] = *max_q_value;
        }
        state_value
    }

    fn train(&mut self, num_episode: u64) {
        // Number of episode the agent has won.
        let mut wins = 0;

        let mut equalities = 0;
        let mut looses = 0;

        for episode in 0..num_episode {
            // We record the trajectory of the episode.
            let mut trajectory = Trajectory::new();

            loop {
                // We observe the environment.
                let (_, observation, _) = self.env.observe();

                let action = self.epsilon_greedy_policy(&observation);

                // We act on the environment.
                self.env.act(action.clone());

                // Record the trajectory.
                let (reward, _, first) = self.env.observe();

                // dbg!((&observation, &action, &reward, first));

                assert!(!trajectory.rewards.contains(&1.0));
                assert!(!trajectory.rewards.contains(&-1.0));
                trajectory.push((observation, action, reward));

                if first {
                    if reward == 1. {
                        wins += 1;
                    } else if reward == -1. {
                        looses += 1;
                    } else {
                        equalities += 1;

                        assert_eq!(reward, 0.0)
                    }
                    break;
                }
            }

            if episode % EPISODE_PRINT == 0 {
                println!("------------------------");
                println!("Episode {}", episode,);
                println!("wins {:.2}%", (wins as f64) / (episode as f64 + 1.) * 100.);
                println!(
                    "Loose {:.2}%",
                    (looses as f64) / (episode as f64 + 1.) * 100.
                );
                println!(
                    "equalities {:.2}%",
                    (equalities as f64) / (episode as f64 + 1.) * 100.
                );
            }
            // At each end of episode we update our action-value function.
            self.update_action_value(trajectory);
        }
    }
}

// Some helper function

// TODO: try to have the same result than matplotlib.
fn _plot_action_value_fn(action_value: &HashMap<(Observation, Action), f64>) {
    let root = BitMapBackend::new("images/3d-surface.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("3D Surface", ("sans-serif", 40))
        .build_cartesian_3d(1.0..10.0, -1.0..1.0, 21.0..1.0)
        .unwrap();

    chart.configure_axes().draw().unwrap();

    // More or less the equivalent of matplotlib `plot_surface`
    // chart.draw_series(SurfaceSeries::xoz(a, b, f));
    chart
        .draw_series(LineSeries::new(
            action_value.iter().map(|((observation, _), reward)| {
                (
                    observation.bank_sum as f64,
                    *reward,
                    observation.player_sum as f64,
                )
            }),
            &RED,
        ))
        .unwrap();
}

// Save the state value function
fn save(state_value: Array2<f64>) {
    let state_value = serde_json::to_value(state_value).unwrap();

    let path = "/home/teddy/dev/python/easy21_plot/value_function.json";

    let mut output = File::create(path).unwrap();
    write!(output, "{}", serde_json::to_string(&state_value).unwrap()).unwrap();
}

fn main() {
    /// Number of episodes we will do to polish our estimation.
    const NUM_EPISODE: u64 = 50_000_000;
    let env = Easy21::default();
    let mut mc_agent = MonteCarloAgent::new(env);
    mc_agent.train(NUM_EPISODE);
    let state_value = mc_agent.compute_state_value_function();

    // println!("{}", state_value);
    save(state_value);
}
