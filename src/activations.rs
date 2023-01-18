//activation functions are pure functions that define how each node/neuron is activated
use tensorflow::ops;
use tensorflow::DataType;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;

//NOTE: this is mainly just used for composing operations with lazy scope assignment and extending
//beyond native TF activation operations
pub type Activation = Option<Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>>;
//TODO: @DEPRECATED: there doesnt seem to be a need to derive or attach traits here but a note
//should be made that such refactors should happen here, not in an implementing scope or crate
//pub struct Activation {
//    pub function: Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>,
//}
pub fn Tanh() -> Activation {
    Some(Box::new(move |output, scope| ops::tanh(output, scope)))
}
pub fn Tan() -> Activation {
    Some(Box::new(move |output, scope| ops::tan(output, scope)))
}
pub fn Sigmoid() -> Activation {
    Some(Box::new(move |output, scope| ops::sigmoid(output, scope)))
}
pub fn Relu() -> Activation {
    Some(Box::new(move |output, scope| ops::relu(output, scope)))
}
pub fn Elu() -> Activation {
    Some(Box::new(move |output, scope| ops::elu(output, scope)))
}
//etc..
