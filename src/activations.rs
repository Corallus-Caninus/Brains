//activation functions are pure functions that define how each node/neuron is activated
use tensorflow::ops;
use tensorflow::DataType;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;

//TODO: refactor this into traits and structs

//@DEPRECATED
//TODO: use trait for layer instead of a function type
//TODO: function types are bad! we have to make do with a structure so we have trait features
///an activation function to be used in a layer.
//pub type Activation = Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>;
//TODO: would prefer dynamic dispatch of a trait object over dyn type here
//pub struct Activation {
//    pub function: Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>,
//}
//pub fn tanh(max_integer: u32) -> Activation {
//    Box::new(move |output, scope| {
//        Ok(ops::multiply(
//            ops::tanh(output, scope)?,
//            ops::constant(max_integer as f32, scope)?,
//            scope,
//        )?
//        .into())
//    })
//}
//TODO: struct with fn pointer
//pub trait Activation {
//    fn function(&self, input: Output, scope: &mut Scope) -> Result<Operation, Status>;
//}
pub struct Activation {
    pub function: Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>,
}
pub fn Tanh() -> Activation {
    Activation {
        function: Box::new(move |output, scope| ops::tanh(output, scope)),
    }
}
pub fn Sigmoid() -> Activation {
    Activation {
        function: Box::new(move |output, scope| ops::sigmoid(output, scope)),
    }
}
pub fn Relu() -> Activation {
    Activation {
        function: Box::new(move |output, scope| Ok(ops::relu(output, scope)?.into())),
    }
}

//fn elu
//etc..
