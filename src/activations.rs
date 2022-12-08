//activation functions are pure functions that define how each node/neuron is activated
use tensorflow::ops;
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
pub fn Tanh(max_integer: u32) -> Activation {
    Activation {
        function: Box::new(move |output, scope| {
            Ok(ops::multiply(
                ops::tanh(output, scope)?,
                ops::constant(max_integer as f32, scope)?,
                scope,
            )?
            .into())
        }),
    }
}
pub fn Sigmoid(max_integer: u32) -> Activation {
    Activation {
        function: Box::new(move |output, scope| {
            Ok(ops::multiply(
                ops::sigmoid(output, scope)?,
                ops::constant(max_integer as f32, scope)?,
                scope,
            )?
            .into())
        }),
    }
}
pub fn Relu() -> Activation {
    Activation {
        function: Box::new(move |output, scope| Ok(ops::relu(output, scope)?.into())),
    }
}
//impl Activation for Tanh {
//    fn function(&self, output: Output, scope: &mut Scope) -> Result<Operation, Status> {
//        Ok(ops::multiply(
//            ops::tanh(output, scope)?,
//            ops::constant(self.max_integer as f32, scope)?,
//            scope,
//        )?
//        .into())
//    }
//}
//pub fn tanh(max_integer: u32) -> Box<Tanh> {
//    Box::new(Tanh { max_integer })
//}
//TOOD: with trait not type alias
//pub fn relu(max_integer: u32) -> Activation {
//    Box::new(move |output, scope| {
//        Ok(ops::multiply(
//            ops::relu(output, scope)?,
//            ops::constant(max_integer as f32, scope)?,
//            scope,
//        )?
//        .into())
//    })
//}
//pub fn sigmoid(max_integer: u32) -> Activation {
//    Box::new(move |output, scope| {
//        Ok(ops::multiply(
//            ops::sigmoid(output, scope)?,
//            ops::constant(max_integer as f32, scope)?,
//            scope,
//        )?
//        .into())
//    })
//}

//fn elu
//etc..
