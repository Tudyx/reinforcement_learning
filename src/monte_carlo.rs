//! Monte Carlo agent.
//! There is no discouting factor for the Easy21 assignement.
//!
//!

// FIXME: in the plot we are not enought close to one.
// TODO: benchmark le passage de Copy a CLone
// TODO: benchmark si jamais je passe des reference pour le lookup.

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

/// Represent a trajectory of through an episode.
// Use a struct of array instead of an array of struct for efficiency.
// TODO: benchmark to be sure its more efficient.
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
    /// Iterate over states and action
    fn iter(&self) -> impl Iterator<Item = (&Observation, &Action)> {
        self.states.iter().zip(self.actions.iter())
    }
}

struct MonteCarloAgent {
    /// Our action-value function (q value)  that we will try to estimate.
    action_value: HashMap<(Observation, Action), f64>,
    /// The N(s) function. Give number of time we have visited a state.
    visited_states: HashMap<Observation, i32>,
    // The N(s,a) function. Give the nimber of time we have visited a couple action state.
    visited_state_action: HashMap<(Observation, Action), u64>,
    /// The underlying environment.
    env: Easy21,
}

impl MonteCarloAgent {
    fn new(env: Easy21) -> Self {
        Self {
            action_value: HashMap::new(),
            visited_states: HashMap::new(),
            visited_state_action: HashMap::new(),
            env,
        }
    }
    /// Update the action-value function from the given trajectory.
    fn update_action_value(&mut self, trajectory: Trajectory) {
        let episode_return = trajectory.cumulated_reward();

        for (state, action) in trajectory.iter() {
            let alpha =
                1. / self.visited_state_action[&(state.to_owned(), action.to_owned())] as f64;

            self.action_value
                .entry((state.clone(), action.clone()))
                .and_modify(|value| *value = *value + alpha * (episode_return - *value))
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

    /// We take a greedy action with a probability of epsilon. Otherwise we take a random action.
    fn epsilon_greedy_policy(&self, observation: &Observation) -> Action {
        // The more we have visited the state, the more epsilon will be small and the more
        // we will take greedy actions (probability 1 - epsilon) (we don't explore). The less we have seen
        // the state, the more we explore.
        let epsilon = N_0 / (N_0 + self.visited_states[&observation] as f64);

        if thread_rng().gen::<f64>() < epsilon {
            self.choose_greedy_action(&observation)
        } else {
            Self::choose_random_action()
        }
    }

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

    fn learn_action_value_fuction(&mut self, num_episode: u64) {
        for _ in 0..num_episode {
            // We record the trajectory of the episode.
            let mut trajectory = Trajectory::new();
            let mut step = 0;

            loop {
                // We observe the environment.
                let (_, observation, first) = self.env.observe();

                self.visited_states
                    .entry(observation.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                let action = self.epsilon_greedy_policy(&observation);

                // We act on the environment.
                self.env.act(action.clone());

                self.visited_state_action
                    .entry((observation.clone(), action.clone()))
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                // Record the trajectory.
                let (reward, _, _) = self.env.observe();
                trajectory.push((observation, action, reward));

                if step > 0 && first {
                    break;
                }
                step += 1;
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
    const NUM_EPISODE: u64 = 100_000;
    let env = Easy21::default();
    let mut mc_agent = MonteCarloAgent::new(env);
    mc_agent.learn_action_value_fuction(NUM_EPISODE);
    let state_value = mc_agent.compute_state_value_function();
    save(state_value);
}
