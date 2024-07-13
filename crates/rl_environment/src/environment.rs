// TODO: replace by bitflag
#[derive(Debug)]
pub enum RenderMode {
    Human,
    RgbArray,
    Ansi,
}

pub enum Observable {
    /// Chess
    Fully,
    /// Poker
    Partially,
}

/// Est-ce qu'une action dans une même configuration dans toujours le même résultat?
/// Certain environment peuvent devenir stochastic selon des parametre
pub enum Foo {
    /// Cart Pole
    Deterministic,
    /// DeepTraffic
    Stochastic,
}

pub enum Bar {
    /// Chess
    Static,
    /// DeepTraffic
    Dynamic,
}
pub enum Fooz {
    /// Chess
    Discrete,
    /// Cart Pole
    Contiguous,
}

#[derive(Debug, Default)]
pub struct Metadata {
    pub render_modes: Vec<RenderMode>,
    pub render_fps: usize,
}

#[derive(Debug)]
pub struct Metadata2<const N: usize> {
    pub render_modes: [RenderMode; N],
    pub render_fps: usize,
}
// A training step. This is generic over the observation.

pub struct Step<O> {
    last_reward: f64,
    observation: O,
    first: bool,
}

impl<O> Step<O> {
    pub fn new(last_reward: f64, observation: O, first: bool) -> Self {
        Self {
            last_reward,
            observation,
            first,
        }
    }

    /// The reward corresponding to the last agent's action. If no agent has acted on
    /// the environment since, then this value is the same.
    pub fn last_reward(&self) -> f64 {
        self.last_reward
    }

    /// Is this step the first one?
    /// For episodic environment this means the beginning of a new episode.
    pub fn is_first(&self) -> bool {
        self.first
    }

    /// Return an observation from the current state of the environment.
    pub fn observation(&self) -> &O {
        &self.observation
    }
}

/// Inspired from [Gym3 API]
///
/// Pros:
/// - `VecEnv` really easy to set up (Classic Gym API Seems to need to duplicate a lot of code to supporte Vec env)
/// - user has no worry about resetting
/// - no undefined behavior after reset
///
/// Cons:
/// - no final observation
/// - counterintuitive gap between observation and reward in the observe method
/// - no differentiation between termination and truncation (https://github.com/openai/gym/issues/2510)
///
/// [Gym3 API]: https://github.com/openai/gym3
pub trait Gym3Environment {
    /// Define the type of the action that an agent can take in the environment.
    /// The domain of validity of this type must be as close as possible as the action space.
    type Action;
    /// Define the type of the observation that an agent can observe in the environment. The observation can represent the full
    /// state of the environment or only on part for environment partially observable.
    /// The domain of validity of this type must be as close as possible as the state space.
    type Observation;

    /// Act on the environment. To see the result of the action you must call `observe` function.
    fn act(&mut self, action: Self::Action);

    /// Observe the current state of the environment. An agent can observe multiple time between 2 actions.
    /// Indeed in case of multi-agent environment, the environment might change between 2 actions.
    fn observe(&self) -> Step<Self::Observation>;
}

// Experiment over https://github.com/openai/gym/issues/2510
#[derive(Clone, Copy)]
enum Done {
    Termination,
    Truncation,
    NotDone,
}
impl From<Done> for bool {
    fn from(value: Done) -> Self {
        match value {
            Done::Termination | Done::Truncation => true,
            Done::NotDone => false,
        }
    }
}
// Pros: more readable. The link betweens truncaation and termination is clearer.
#[derive(Clone, Copy)]
enum Done2 {
    True(Reason),
    False,
}
#[derive(Clone, Copy)]
enum Reason {
    Termination,
    Truncation,
}
impl From<Done2> for bool {
    fn from(value: Done2) -> Self {
        match value {
            Done2::True(_) => true,
            Done2::False => false,
        }
    }
}

#[test]
fn done_api() {
    /////// intialization ///////
    let done = Done::Termination;
    let done2 = Done2::True(Reason::Termination); // More verbose (but's its env side)

    /////// If condition ///////
    // equality
    if done.into() {
        println!("Episode over");
    }
    if done2.into() {
        println!("Episode over");
    }

    /////// match ///////
    match done {
        Done::Termination => {
            println!("Do something special about termination and break the episode")
        }
        Done::Truncation => println!("Do something special about truncation and break the episode"),
        Done::NotDone => println!("continue"),
    };

    match done2 {
        Done2::True(reason) => {
            match reason {
                Reason::Termination => println!("Do something special about termination"),
                Reason::Truncation => println!("Do something special about truncation"),
            }
            println!("break the episode"); // Code better factorized.
        }
        Done2::False => println!("continue"),
    }

    /////// DQN potential use case ///////
    if matches!(done, Done::Termination) {
        println!("next_state = None");
    } else {
        println!("next_state = torch.tensor(observation, ...");
    }

    // more verbose
    if matches!(done2, Done2::True(Reason::Termination)) {
        println!("next_state = None");
    } else {
        println!("next_state = torch.tensor(observation, ...");
    }
}
