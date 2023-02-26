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
/// - `VecEnv` really easy to set up
/// - user has no worry about resetting
/// - no undefined behavior after reset
///
/// Cons:
/// - no final observation
/// - counterintuitive gap between observation and reward in the observe method
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
