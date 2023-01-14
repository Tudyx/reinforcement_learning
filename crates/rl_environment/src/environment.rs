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

struct Observe<O> {
    last_reward: f64,
    observation: O,
    first: bool,
}

pub type Reward = f64;

// inspired de gym3 API.
// la gym3 api ne gere pas les terminated observation.
// y'a un décalage contre-intuitif dans l'observation.
pub trait Gym3Environment {
    // const METADATA: Metadata2<2>;
    type Action;
    type Observation;

    // TODO: the last tuple element is a dict
    fn act(&mut self, action: Self::Action);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> (Reward, Self::Observation, bool);
}