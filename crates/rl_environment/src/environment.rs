// TODO: replace by bitflag
#[derive(Debug)]
pub enum RenderMode {
    Human,
    RgbArray,
    Ansi,
}

pub enum Osbservable {
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

pub trait Space<T> {}

/// Je pense que l'action space est utile pour l'agent.

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
    /// the environment, then this value don't change.
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

pub type Reward = f64;

// inspired de gym3 API.
// la gym3 api ne gere pas les terminated observation.
// y'a un décalage contre-intuitif dans l'observation.
pub trait Gym3Environment {
    /// Define the type of the action that an agent can take in the environment.
    /// The domaine of validity of this type must be as close as possible as the action space.
    type Action;
    /// Define the type of the observation that an agent can observe in the environment. The observation can represent the full
    /// state of the environment or only on part for environment paratially observable.
    /// The domaine of validity of this type must be as close as possible as the state space.
    type Observation;

    // TODO: the last tuple element is a dict
    fn act(&mut self, action: Self::Action);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> Step<Self::Observation>;
}
