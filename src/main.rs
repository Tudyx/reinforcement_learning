// In python you can portailly update that's why environment initialization is divided into the constructor and the reset function.
// They choose to use an array of 2 elements to represent the cordonate in an array of 2 dimension. That's probably because they want to use tensor later.

// TODO: manage la seed.

// open question
// - Est-ce que je peux utiliser de trucs plus idiomatque pour les enum? Je pourrait faire des "From"
// - Est ce que l'action space je peux le definir à partir d'un associated constant?
// - Est ce que je fois parallelizer l'environment pour voir si cela raoute des contarintes?
mod algorithm;
mod easy_21;
mod environment;
pub use environment::Environment;

mod grid_word;
mod viewer;
mod wrapper;

fn main() {
    println!("Hello world");
}
