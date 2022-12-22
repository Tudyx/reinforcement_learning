use crate::environment::Environment;


// TODO: introduce egui to this

struct ViewerWraper<E>
where
    E: Environment,
{
    env: E,
    width: u64,
    height: u64,
}

impl<E> Environment for ViewerWraper<E>
where
    E: Environment,
{
    type Action = E::Action;

    type Observation = E::Observation;

    fn act(&mut self, action: Self::Action) {
        self.env.act(action);
        let image = self.get_image();
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        self.env.observe()
    }

    fn metadata() -> crate::environment::Metadata {
        E::metadata()
    }
}


impl<E> ViewerWraper<E> where E: Environment {
    fn get_image(&self){}
}
