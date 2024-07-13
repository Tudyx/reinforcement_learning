#[cfg(test)]
mod tests {
    use crate::easy_21::Action;
    use crate::easy_21::Easy21;
    use crate::environment::VecEnv;
    use crate::environment::VecEnvironment;

    #[test]
    #[should_panic]
    fn wrong_action_number() {
        let env = Easy21::default();

        let mut envs = VecEnv::new(env, 5);
        // we provid 0 action but we have 5 environment: panic!
        envs.act(vec![]);
    }

    #[test]
    fn vec_env_basics() {
        let env = Easy21::default();

        let mut envs = VecEnv::new(env, 5);

        dbg!(envs.observe());

        envs.act(vec![
            Action::Hit,
            Action::Hit,
            Action::Hit,
            Action::Hit,
            Action::Stick,
        ]);

        dbg!(envs.observe());
    }
}
