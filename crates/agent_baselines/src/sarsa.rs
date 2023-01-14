//! SARSA(lambda) agent using backward view.
//!
//! Is applying TD(lambda) backward view to the control problem.
//!
//! The backward view is used to allow the algorithm to be on-policy (updated on each step).
//! For that, egibility traces are used, which give more credit to the state we have seen more recently and more frequently.
//!

use environment_baselines::easy_21::{Action, Easy21, Observation};
use itertools::izip;
use ndarray::{Array, Array2};
use rand::prelude::*;
use rl_environment::{Gym3Environment, Step};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Define how much we explore.
const N_0: f64 = 100.;
/// Print status every x episode.
const EPISODE_PRINT: u64 = 1000;

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

    /// Iter on the trajectory in the reverse order.
    fn iter_rev(&self) -> impl Iterator<Item = (&Observation, &Action, &f64)> {
        izip!(&self.states, &self.actions, &self.rewards).rev()
    }
}

/// An agent using Monte Carlo control to find the best policy.
struct TdAgent {
    /// Our action-value function (Q value) that we will try to improve towards Q*(s, a).
    action_value: HashMap<(Observation, Action), f64>,
    /// The N(s) function. Give number of time we have visited a state.
    visited_states: HashMap<Observation, i32>,
    /// The N(s,a) function. Give the number of time we have visited a couple action state.
    visited_state_action: HashMap<(Observation, Action), u64>,
    // The E(s,a) describe how much we should update the action-value.
    // The more we have seen the state and the more recent it is, the more the eligibity trace will be big.
    // We consider that state we have seen more recently and more frequently deserve to be more updated by the reward.
    eligibility_traces: HashMap<(Observation, Action), f64>,
    /// The underlying environment.
    env: Easy21,
    /// [0; 1] the more lamnda is big, the more we will give importance to past state in the reward contribution.
    lambda: f64,
}

impl TdAgent {
    fn new(env: Easy21, lambda: f64) -> Self {
        Self {
            action_value: HashMap::with_capacity(10 * 21 * 2),
            visited_states: HashMap::with_capacity(10 * 21),
            visited_state_action: HashMap::with_capacity(10 * 21 * 2),
            eligibility_traces: HashMap::with_capacity(10 * 21 * 2),
            env,
            lambda,
        }
    }

    // alpha, A.K.A.the step size.
    fn compute_step_size(&self, state: &Observation, action: &Action) -> f64 {
        1. / self.visited_state_action[&(state.to_owned(), action.to_owned())] as f64
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
            .get(&(*observation, Action::Stick))
            .unwrap_or(&0.);

        let hit_value = self
            .action_value
            .get(&(*observation, Action::Hit))
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
            self.choose_greedy_action(observation)
        }
    }

    /// We compute the estimated state value function from the estimated action value function.
    fn compute_state_value_function(&self) -> Array2<f64> {
        let mut state_value = Array::zeros((21, 10));

        for observation in self.visited_states.keys() {
            let stick_value = self
                .action_value
                .get(&(*observation, Action::Stick))
                .unwrap_or(&0.);

            let hit_value = self
                .action_value
                .get(&(*observation, Action::Hit))
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

    fn compute_delta(
        &mut self,
        reward: f64,
        next_observation: Observation,
        next_action: Action,
        observation: Observation,
        action: Action,
    ) -> f64 {
        reward
            + *self
                .action_value
                .entry((next_observation, next_action))
                .or_insert(0.)
            - *self.action_value.entry((observation, action)).or_insert(0.)
    }

    fn train(&mut self, num_episode: u64) {
        // Number of episode the agent has won.
        let mut wins = 0;

        let mut equalities = 0;
        let mut looses = 0;

        for episode in 0..num_episode {
            // We clear eligibility traces.
            self.eligibility_traces.clear();
            loop {
                let observation = *self.env.observe().observation();
                let action = self.epsilon_greedy_policy(&observation);
                self.env.act(action);
                let step = self.env.observe();

                // The next action we will take if we follow our policy.
                let next_action = self.epsilon_greedy_policy(step.observation());

                self.visited_state_action
                    .entry((observation, action))
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                self.visited_states
                    .entry(observation)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                // TD-error δ. Estimated value minus the one we really got.
                let delta = self.compute_delta(
                    step.last_reward(),
                    *step.observation(),
                    next_action,
                    observation,
                    action,
                );

                // E(S, A) <- E(S, A) + 1
                self.eligibility_traces
                    .entry((observation, action))
                    .and_modify(|count| *count += 1.)
                    .or_insert(1.);

                // Step size α
                let alpha = self.compute_step_size(&observation, &action);

                for (state, action) in self.visited_state_action.keys() {
                    // For each state action pair we update the value with αδE(s, a)
                    self.action_value
                        .entry((*state, *action))
                        .and_modify(|value| {
                            *value += alpha
                                * delta
                                * *self
                                    .eligibility_traces
                                    .entry((*state, *action))
                                    .or_insert(0.0)
                        })
                        .or_insert(0.);

                    let eligibility_trace = self.eligibility_traces[&(*state, *action)];

                    // At each step, we decrease the value of all states. More recent state will have more importance.
                    // IndexMut is not implemented for HashMap, hence the get_mut unwrap.
                    *self.eligibility_traces.get_mut(&(*state, *action)).unwrap() *= self.lambda;
                }

                if step.is_first() {
                    if step.last_reward() == 1. {
                        wins += 1;
                    } else if step.last_reward() == -1. {
                        looses += 1;
                    } else {
                        equalities += 1;

                        debug_assert_eq!(step.last_reward(), 0.0)
                    }
                    break;
                }
            }

            if episode % EPISODE_PRINT == 0 && episode != 0 {
                println!("------------------------");
                println!("lambda = {:.1}", self.lambda);
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
        }
    }
}

// Save the state value function
fn save(state_value: Array2<f64>) {
    let state_value = serde_json::to_value(state_value).unwrap();

    let path = "/home/teddy/dev/python/easy21_plot/value_function.json";

    let mut output = File::create(path).unwrap();
    write!(output, "{}", serde_json::to_string(&state_value).unwrap()).unwrap();
}

fn mean_square_error(
    q_star: &HashMap<(Observation, Action), f64>,
    q: &HashMap<(Observation, Action), f64>,
) -> f64 {
    let mut mean_square_error = 0.;
    for ((state, action), q_value) in q_star.iter() {
        mean_square_error += (q_value - q.get(&(*state, *action)).unwrap_or(&0.)).powf(2.);
    }
    mean_square_error
}

fn main() {
    /// Number of episodes we will do to polish our estimation.
    const NUM_EPISODE: u64 = 300_000;

    let lambda = 0.8;
    let env = Easy21::default();
    let mut td_agent = TdAgent::new(env, lambda);
    td_agent.train(NUM_EPISODE);

    let q_star = td_agent.action_value;

    let mut mean_square_errors = Vec::new();
    // for charts
    let mut lambdas = Vec::new();

    for lambda in 0..=10 {
        let lambda = f64::from(lambda) * 0.1;
        lambdas.push(lambda);
        let env = Easy21::default();
        let mut td_agent = TdAgent::new(env, lambda);
        td_agent.train(1000);
        mean_square_errors.push(mean_square_error(&q_star, &td_agent.action_value));
    }

    println!("{lambdas:?}");
    println!("{mean_square_errors:?}");
}
