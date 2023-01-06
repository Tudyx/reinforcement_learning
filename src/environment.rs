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
// y'a un décalage contre intuitif dans l'observation.
pub trait Gym3Environment {
    // const METADATA: Metadata2<2>;
    type Action;
    type Observation;

    // TODO: the last tuple element is a dict
    fn act(&mut self, action: Self::Action);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> (Reward, Self::Observation, bool);

    // Associated constant can't use const generics yet, that what we can't change render_modes to an array an
    // use associated constant.
    fn metadata() -> Metadata;
}

pub trait VecEnvironment {
    type Action;
    type Observation;

    fn act(&mut self, action: Vec<Self::Action>);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> Vec<(Reward, Self::Observation, bool)>;
}

pub struct VecEnv<E>
where
    E: Gym3Environment,
{
    envs: Vec<E>,
}

// TODO: tester avec une const parameter et un array
impl<E> VecEnvironment for VecEnv<E>
where
    E: Gym3Environment + Clone,
{
    type Action = E::Action;
    type Observation = E::Observation;

    fn act(&mut self, actions: Vec<Self::Action>) {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "You must provide the same num of actions that the number of envs."
        );

        for (action, env) in actions.into_iter().zip(self.envs.iter_mut()) {
            env.act(action);
        }
    }
    fn observe(&self) -> Vec<(Reward, Self::Observation, bool)> {
        self.envs.iter().map(|env| env.observe()).collect()
    }
}

impl<E> VecEnv<E>
where
    E: Gym3Environment + Clone,
{
    // Create a new vec env.
    pub fn new(env: E, nb_env: usize) -> Self {
        Self {
            envs: vec![env; nb_env],
        }
    }
}
