use ndarray::{s, Array, Array2, Array3};
use plotters::prelude::*;
use rand::prelude::*;
use rust_gym::easy_21::{Action, Easy21, Observation};
use rust_gym::Environment;
use std::collections::HashMap;

/// Retex:
/// Ord n'est pas implémenter pour f64!

const NUM_EPISODE: usize = 1_000;

const N0: f64 = 100.;

// There is no discouting factor for the Easy21 environment

fn plot_action_value_fn(action_value: &HashMap<(Observation, Action), f64>) {
    let root = BitMapBackend::new("images/3d-surface.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("3D Surface", ("sans-serif", 40))
        .build_cartesian_3d(1.0..10.0, -1.0..1.0, 21.0..1.0)
        .unwrap();

    chart.configure_axes().draw().unwrap();
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

struct MonteCarloAgent {
    /// Our action-value function (q value)  that we will try to estimate.
    action_value: HashMap<(Observation, Action), f64>,
    env: Easy21,
}

impl MonteCarloAgent {
    fn new(env: Easy21) -> Self {
        Self {
            action_value: HashMap::new(),
            env,
        }
    }
    fn random_policy() -> Action {
        if random() {
            Action::Stick
        } else {
            Action::Hit
        }
    }
    fn greedy_policy(&self, observation: &Observation) -> Action {
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

    fn learn_action_value_fuction(&mut self) {}
}
fn main() {
    let mut env = Easy21::default();

    // The N(s) function. Give number of time we have visited a state.
    let mut visited_states = HashMap::new();
    // The N(s,a) function. Give the nimber of time we have visited a couple action state.
    let mut visited_state_action = HashMap::new();

    // Our action value funnction. For a state and an action, give the expected return. Initialized with zero.
    let mut action_value = HashMap::new();

    for _ in 0..NUM_EPISODE {
        // We record the trajectory of the episode.
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();

        // FIXME: change it to an act/observe instead of observe/act

        let mut step = 0;

        loop {
            // We observe the environment.
            let (_, observation, first) = env.observe();

            visited_states
                .entry(observation.clone())
                .and_modify(|count| *count += 1)
                .or_insert(1);

            // The more we have visited the state, the more epsilon will be small and the more
            // we will take greedy actions (probability 1 - epsilon) (we don't explore). The less we have seen
            // the state, the more we explore.
            let epsilon = N0 / (N0 + visited_states[&observation] as f64);
            let exploring_prob = 1. - epsilon;

            let action = if thread_rng().gen::<f64>() < exploring_prob {
                // We choose a random action.
                if random() {
                    Action::Stick
                } else {
                    Action::Hit
                }
            } else {
                // We choose the geedy value (we take the action with the maximum action value).
                let stick_value = action_value
                    .get(&(observation.clone(), Action::Stick))
                    .unwrap_or(&0.);

                let hit_value = action_value
                    .get(&(observation.clone(), Action::Hit))
                    .unwrap_or(&0.);

                if stick_value > hit_value {
                    Action::Stick
                } else {
                    Action::Hit
                }
            };

            // We act on the environment.
            env.act(action.clone());

            visited_state_action
                .entry((observation.clone(), action.clone()))
                .and_modify(|count| *count += 1)
                .or_insert(1);

            // Record the trajectory.
            let (reward, _, _) = env.observe();
            states.push(observation);
            actions.push(action);
            rewards.push(reward);

            if step > 0 && first {
                break;
            }
            step += 1;
        }

        let episode_return = rewards.iter().sum::<f64>();
        // We iter over the trajectory.
        for step in 0..rewards.len() {
            let state = &states[step];
            let action = &actions[step];

            let alpha = 1. / visited_state_action[&(state.to_owned(), action.to_owned())] as f64;

            action_value
                .entry((state.clone(), action.clone()))
                .and_modify(|value| *value = *value + alpha * (episode_return - *value))
                .or_insert(0.);
        }

        // We update our value function at the end of each episode
        // TODO: iterate over the trajectory, update `action_value`
    }
    let mut state_value = Array::zeros((21, 10));

    for (observation, _) in visited_states {
        // We compute q max

        let stick_value = action_value
            .get(&(observation.clone(), Action::Stick))
            .unwrap_or(&0.);

        let hit_value = action_value
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

    save(state_value);

    // plot_action_value_fn(&action_value);
}

use std::fs::File;
use std::io::Write;
// Save the state value function
fn save(state_value: Array2<f64>) {
    let state_value = serde_json::to_value(state_value).unwrap();

    let path = "/home/teddy/dev/python/easy21_plot/value_function.json";

    let mut output = File::create(path).unwrap();
    write!(output, "{}", serde_json::to_string(&state_value).unwrap()).unwrap();
}
