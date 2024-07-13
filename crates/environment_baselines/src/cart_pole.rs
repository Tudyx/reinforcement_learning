use rand::distributions::Uniform;
use rand::prelude::*;
use rl_environment::{Gym3Environment, Step};

pub enum Action {
    PushCartToTheLeft,
    PushCartToTheRight,
}

/// Reward:
/// Reward is 1 for every step taken, including the termination step.
#[derive(Debug)]
pub struct CartPole {
    /// The state of the environment, what the agent observe.
    state: State,
    /// Different parameters of the cart pole environment.
    pub config: Config,
    /// Cumulated reward during the episode.
    episodic_return: f64,
    /// If the number of step during one episode. If its greater than 500 then it end the episode. (Truncation)
    episodic_length: u32,
    /// The random number generator used to intialize the state.
    rng: StdRng,
    first: bool,
}

/// What the agent observe. Here the observation equal
/// the environment state, its a fully observable environment.
#[derive(Copy, Clone, Debug)]
pub struct State {
    /// [-4.8; 4.8].
    cart_position: f64,
    /// [-inf; inf].
    cart_velocity: f64,
    /// [-0.418 rad (-24°);0.418 rad (24°)].
    pole_angle: f64,
    /// [-inf; inf].
    pole_angular_velocity: f64,
}

impl Default for CartPole {
    fn default() -> Self {
        let mut rng = StdRng::from_entropy();
        let distr = Uniform::new(-0.05, 0.05);

        let state = State {
            cart_position: rng.sample(distr),
            cart_velocity: rng.sample(distr),
            pole_angle: rng.sample(distr),
            pole_angular_velocity: rng.sample(distr),
        };

        Self {
            state,
            config: Default::default(),
            episodic_return: 0.,
            episodic_length: 0,
            rng,
            first: true,
        }
    }
}

#[derive(Debug)]
pub enum KinematicsIntegrator {
    Euler,
    SemiImplicitEuler,
}

#[derive(Debug)]
pub struct Config {
    pub cart_mass: f64,
    pub pole_mass: f64,
    pub force_mag: f64,
    pub pole_mass_length: f64,
    pub total_mass: f64,
    /// Gravity in m/s^2. 9.8 on earth.
    pub gravity: f64,
    pub half_pole_length: f64,
    pub kinematics_integrator: KinematicsIntegrator,
    /// Seconds between state updates.
    pub tau: f64,
    /// Episode end if the cart position is more than this threshold. (Termination)
    pub x_threshold: f64,
    /// Angle at which to fail the episode. (Termination)
    pub theta_threshold_radians: f64,
}

impl Default for Config {
    fn default() -> Self {
        let cart_mass = 1.0;
        let pole_mass = 0.1;
        let half_pole_length = 0.5;
        Self {
            cart_mass,
            pole_mass,
            force_mag: 10.0,
            pole_mass_length: pole_mass * half_pole_length,
            total_mass: cart_mass + pole_mass,
            gravity: 9.8,
            half_pole_length,
            kinematics_integrator: KinematicsIntegrator::Euler,
            tau: 0.02,
            x_threshold: 2.4,
            theta_threshold_radians: 12.0 * 2.0 * std::f64::consts::PI / 360.0,
        }
    }
}

impl Gym3Environment for CartPole {
    type Action = Action;

    type Observation = State;

    fn act(&mut self, action: Self::Action) {
        if self.first {
            self.first = false;
        }

        let mut x = self.state.cart_position;
        let mut x_dot = self.state.cart_velocity;
        let mut theta = self.state.pole_angle;
        let mut theta_dot = self.state.pole_angular_velocity;

        let force = match action {
            Action::PushCartToTheLeft => self.config.force_mag,
            Action::PushCartToTheRight => -self.config.force_mag,
        };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // For the intersted reader:
        // https://coneural.org/florian/papers/05_cart_pole.pdf
        let temp = (force + self.config.pole_mass_length * theta_dot.powi(2) * sin_theta)
            / self.config.total_mass;
        let theta_acc = (self.config.gravity * sin_theta - cos_theta * temp)
            / (self.config.half_pole_length
                * (4.0 / 3.0 - self.config.pole_mass * cos_theta.powi(2) / self.config.total_mass));
        let x_acc =
            temp - self.config.pole_mass_length * theta_acc * cos_theta / self.config.total_mass;

        match self.config.kinematics_integrator {
            KinematicsIntegrator::Euler => {
                x += self.config.tau * x_dot;
                x_dot += self.config.tau * x_acc;
                theta += self.config.tau * theta_dot;
                theta_dot += self.config.tau * theta_acc;
            }
            KinematicsIntegrator::SemiImplicitEuler => {
                x_dot += self.config.tau * x_acc;
                x += self.config.tau * x_dot;
                theta_dot += self.config.tau * theta_acc;
                theta += self.config.tau * theta_dot;
            }
        }

        self.state = State {
            cart_position: x,
            cart_velocity: x_dot,
            pole_angle: theta,
            pole_angular_velocity: theta_dot,
        };

        self.episodic_return += 1.0;
        self.episodic_length += 1;

        let done: bool = x < -self.config.x_threshold
            || x > self.config.x_threshold
            || theta < -self.config.theta_threshold_radians
            || theta > self.config.theta_threshold_radians
            || self.episodic_length >= 500;

        if done {
            self.reset();
        }
        // TODO: episodic_return and episodic_length where returned here as info.
    }

    fn observe(&self) -> Step<Self::Observation> {
        Step::new(1.0, self.state, self.first)
    }
}

impl CartPole {
    fn reset(&mut self) {
        self.first = true;
        self.episodic_length = 0;
        self.episodic_return = 0.0;

        let distr = Uniform::new(-0.05, 0.05);

        self.state = State {
            cart_position: self.rng.sample(distr),
            cart_velocity: self.rng.sample(distr),
            pole_angle: self.rng.sample(distr),
            pole_angular_velocity: self.rng.sample(distr),
        }
    }
}
