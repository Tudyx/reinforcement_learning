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

/// Number of episodes we will do to polish our estimation.
const NUM_EPISODE: usize = 100_000;

/// Define how much we explore.
const N_0: f64 = 100.;

struct MonteCarloAgent {
    /// Our action-value function (q value)  that we will try to estimate.
    action_value: HashMap<(Observation, Action), f64>,
    /// The N(s) function. Give number of time we have visited a state.
    visited_states: HashMap<Observation, i32>,
    /// The underlying environment.
    env: Easy21,
}

impl MonteCarloAgent {
    fn new(env: Easy21) -> Self {
        Self {
            action_value: HashMap::new(),
            visited_states: HashMap::new(),
            env,
        }
    }

    fn choose_random_action() -> Action {
        if random() {
            Action::Stick
        } else {
            Action::Hit
        }
    }

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

    fn epsilon_greedy_policy(&self, observation: &Observation) -> Action {
        // The more we have visited the state, the more epsilon will be small and the more
        // we will take greedy actions (probability 1 - epsilon) (we don't explore). The less we have seen
        // the state, the more we explore.
        let epsilon = N_0 / (N_0 + self.visited_states[&observation] as f64);
        let exploring_prob = 1. - epsilon;

        if thread_rng().gen::<f64>() < exploring_prob {
            Self::choose_random_action()
        } else {
            self.choose_greedy_action(&observation)
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

    fn learn_action_value_fuction(&mut self) {
        // The N(s,a) function. Give the nimber of time we have visited a couple action state.
        let mut visited_state_action = HashMap::new();

        for _ in 0..NUM_EPISODE {
            // We record the trajectory of the episode.
            let mut states = Vec::new();
            let mut actions = Vec::new();
            let mut rewards = Vec::new();

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

                visited_state_action
                    .entry((observation.clone(), action.clone()))
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                // Record the trajectory.
                let (reward, _, _) = self.env.observe();
                states.push(observation);
                actions.push(action);
                rewards.push(reward);

                if step > 0 && first {
                    break;
                }
                step += 1;
            }

            // At each end of episode we update or action value.

            let episode_return = rewards.iter().sum::<f64>();
            // We iter over the trajectory.
            for step in 0..rewards.len() {
                let state = &states[step];
                let action = &actions[step];

                let alpha =
                    1. / visited_state_action[&(state.to_owned(), action.to_owned())] as f64;

                self.action_value
                    .entry((state.clone(), action.clone()))
                    .and_modify(|value| *value = *value + alpha * (episode_return - *value))
                    .or_insert(0.);
            }
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
    let env = Easy21::default();
    let mut mc_agent = MonteCarloAgent::new(env);
    mc_agent.learn_action_value_fuction();
    let state_value = mc_agent.compute_state_value_function();
    save(state_value);
}
