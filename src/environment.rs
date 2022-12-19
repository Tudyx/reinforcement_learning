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

#[derive(Debug)]
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
    observation: O,
    reward: f64
}
pub trait Environment {
    // const METADATA: Metadata2<2>;

    // const LOL: usize;

    // type ActionSpace: Space<Self::Action>;
    // type ObservationSpace: Space<Self::Observation>;

    // type RewardRange;
    // type Spec;
    // type NpRandom;

    type Action;
    type Observation;

    // TODO: the last tuple element is a dict
    fn act(&mut self, action: Self::Action);


    fn observe(&self) -> (f64, Self::Observation, bool);


    fn render(&self) {}

    fn close(&mut self) {}

    // Associated constant can't use const generics yet, that what we can't change render_modes to an array an
    // use associated constant.
    fn metadata() -> Metadata;
}
