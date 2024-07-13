use std::io;

use anyhow::bail;

use environment_baselines::easy_21::{Action, Easy21};
use rl_environment::Gym3Environment;

struct Interactive {
    env: Easy21,
    total_steps: usize,
    episode_steps: usize,
}

impl Interactive {
    fn new(env: Easy21) -> Self {
        Self {
            env,
            total_steps: 0,
            episode_steps: 0,
        }
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let observation = *self.env.observe().observation();
        dbg!(observation);
        loop {
            self.update()?;
        }
    }

    fn retrieve_key_pressed(&self) -> anyhow::Result<<Easy21 as Gym3Environment>::Action> {
        let mut action = String::new();
        io::stdin().read_line(&mut action)?;
        let action = action.trim().parse::<usize>()?;
        match action {
            0 => Ok(Action::Hit),
            1 => Ok(Action::Stick),
            _ => bail!("Wrong key!"),
        }
    }

    fn update(&mut self) -> anyhow::Result<()> {
        println!("Enter action:");
        let action = self.retrieve_key_pressed()?;
        let first = self.act(action);

        if first {
            self.episode_steps = 0;
        }

        Ok(())
    }
    fn act(&mut self, action: <Easy21 as Gym3Environment>::Action) -> bool {
        self.env.act(action);

        self.episode_steps += 1;
        self.total_steps += 1;
        let step = self.env.observe();
        println!("Reward: {}", step.last_reward());
        if step.is_first() {
            println!("Finish the episode in {} steps", self.episode_steps);
        }
        dbg!(step.observation());
        step.is_first()
    }
}

fn main() -> anyhow::Result<()> {
    let mut agent = Interactive::new(Easy21::default());
    agent.run()?;
    Ok(())
}
