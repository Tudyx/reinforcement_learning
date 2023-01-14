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

    // Associated constant can't use const generics yet, that what we can't change render_modes to an array an
    // use associated constant.
    fn metadata() -> Metadata;
}

pub trait ArrayEnvironment<const N: usize> {
    type Action;
    type Observation;

    fn act(&mut self, actions: [Self::Action; N]);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> [(Reward, Self::Observation, bool); N];
}

pub trait VecEnvironment {
    type Action;
    type Observation;

    fn act(&mut self, actions: Vec<Self::Action>);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> Vec<(Reward, Self::Observation, bool)>;
}

/// Produce observation on array, no heap allocation here.
pub struct ArrayEnv<E, const N: usize>
where
    E: Gym3Environment,
{
    envs: [E; N],
}

pub struct VecEnv<E>
where
    E: Gym3Environment,
{
    envs: Vec<E>,
}

impl<E, const N: usize> ArrayEnvironment<N> for ArrayEnv<E, N>
where
    E: Gym3Environment,
{
    type Action = E::Action;

    type Observation = E::Observation;

    fn act(&mut self, actions: [Self::Action; N]) {
        for (action, env) in actions.into_iter().zip(self.envs.iter_mut()) {
            env.act(action);
        }
    }

    fn observe(&self) -> [(Reward, Self::Observation, bool); N] {
        // TODO: Check the warning in Array::map about large array concern this function.
        let mut env = self.envs.iter();
        [(); N].map(|_| env.next().expect("we know the size of the env").observe())
    }
}

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
            "You must provide the same number of actions that the number of envs."
        );

        for (action, env) in actions.into_iter().zip(self.envs.iter_mut()) {
            env.act(action);
        }
    }
    fn observe(&self) -> Vec<(Reward, Self::Observation, bool)> {
        self.envs.iter().map(|env| env.observe()).collect()
    }
}

/// Blanket implementation for all Gym3Environment.
impl<E> VecEnv<E>
where
    E: Gym3Environment + Clone,
{
    //FIXME: if we want to initialize with different seed each env?
    // Create a new vec env.
    pub fn new(env: E, nb_env: usize) -> Self {
        Self {
            envs: vec![env; nb_env],
        }
    }
}
