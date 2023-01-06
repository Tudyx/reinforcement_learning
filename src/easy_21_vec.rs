use crate::easy_21::Easy21;
use crate::environment::VecEnv;
use crate::environment::VecEnvironment;

#[test]
fn basics() {
    let env = Easy21::default();

    let mut envs = VecEnv::new(env, 5);

    dbg!(envs.observe());

    envs.act(vec![]);
}
