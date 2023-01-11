use tensorflow::ops;
use tensorflow::train::*;
use tensorflow::DataType;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::TensorType;

//LOSS FUNCTIONS
//TODO: structure the args since label and output is concrete
///A function that will be calculated on the output of the network and the given labels to determine the loss
pub type LossFunction = Box<dyn Fn(&mut Scope, &Output, &Operation) -> Result<Operation, Status>>;

///The l2 loss is the sum of the squared differences between the output and the label.
pub fn l2() -> LossFunction {
    Box::new(|scope: &mut Scope, Output: &Output, Label: &Operation| {
        let Error = ops::square(
            ops::abs(
                ops::sub(
                    Output.clone(),
                    Label.clone(),
                    &mut scope.with_op_name("error"),
                )?,
                scope,
            )?,
            scope,
        );
        Error
    })
}
///The l1 loss is the sum of the absolute differences between the output and the label.
pub fn l1() -> LossFunction {
    Box::new(|scope: &mut Scope, Output: &Output, Label: &Operation| {
        let Error = ops::abs(
            ops::sub(
                Output.clone(),
                Label.clone(),
                &mut scope.with_op_name("error"),
            )?,
            scope,
        );
        Error
    })
}

//OPTIMIZERS//
//TODO: this is only a trait obj because optimizer isnt clone for method chaining
///An optimization strategy for the loss function
pub type LossOptimizer = Box<dyn Optimizer>;

//TODO: need to propagate errors: get rid of all unwraps by looking up how to properly generate
//Tensorflow's error types

//Gradient Descent//
///Builder function for the GradientDescentOptimizer
pub struct GradientBuilder<T>
where
    T: TensorType,
{
    pub learning_rate: Option<T>,
    pub dtype: Option<DataType>,
}
//TODO: these should probably use the builder derive macro crate
///GradientDescent is an optimization algorithm that minimizes a loss function by iteratively moving parameters in the direction of steepest descent as defined by the negative of the gradient of the loss function.
pub fn GradientDescent<T>() -> GradientBuilder<T>
where
    T: TensorType,
{
    GradientBuilder {
        learning_rate: None,
        dtype: None,
    }
}
impl<T> GradientBuilder<T>
where
    T: TensorType,
{
    ///Set the learning rate for the optimizer
    pub fn learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }
    ///Set the data type for the optimizer
    pub fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    ///Build the optimizer to pass to a BrainBuilder
    pub fn build(mut self, scope: &mut Scope) -> Result<LossOptimizer, Status> {
        Ok(Box::new(GradientDescentOptimizer::new(ops::constant(
            self.learning_rate.unwrap(),
            scope,
        )?)))
    }
}

//Adadelta//
///Builder function for the AdadeltaOptimizer
pub struct AdadeltaBuilder<T>
where
    T: TensorType,
{
    pub learning_rate: Option<T>,
    pub rho: Option<T>,
    pub epsilon: Option<T>,
    pub dtype: Option<DataType>,
}
pub fn Adadelta<T>() -> AdadeltaBuilder<T>
where
    T: TensorType,
{
    AdadeltaBuilder {
        learning_rate: None,
        rho: None,
        epsilon: None,
        dtype: None,
    }
}
impl<T> AdadeltaBuilder<T>
where
    T: TensorType,
{
    ///Set the learning rate for the optimizer
    pub fn learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }
    ///Set the rho for the optimizer
    pub fn rho(mut self, rho: T) -> Self {
        self.rho = Some(rho);
        self
    }
    ///Set the epsilon for the optimizer
    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = Some(epsilon);
        self
    }
    ///Set the data type for the optimizer
    pub fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    ///Build the optimizer to pass to a BrainBuilder
    pub fn build(mut self, scope: &mut Scope) -> Result<LossOptimizer, Status> {
        let mut Adadelta = AdadeltaOptimizer::new();
        Adadelta.set_learning_rate(ops::constant(self.learning_rate.unwrap(), scope)?);
        Adadelta.set_rho(ops::constant(self.rho.unwrap(), scope)?);
        Adadelta.set_epsilon(ops::constant(self.epsilon.unwrap(), scope)?);
        Ok(Box::new(Adadelta))
    }
}
