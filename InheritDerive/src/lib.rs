///a macro for deriving Inherited state traits in layer.rs to bring some traits up to the top level
///via accessor and setter blanket implementations
//pub struct T_layer(LayerState);
//impl InheritState for T_Layer{
//    fn get_mut_layer_state(&mut self) -> &mut LayerState {
//        &mut self.0
//    }
//    fn get_layer_state(&self) -> &LayerState {
//        &self.0
//    }
//}
//a trait that automates deriving the InheritState trait for a struct that contains a LayerState as
//self.0
use proc_macro::TokenStream;
extern crate proc_macro;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};
#[proc_macro_derive(InheritState)]
pub fn derive_inherit_state(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let gen = quote! {
        impl InheritState for #name {
            fn get_mut_layer_state(&mut self) -> &mut LayerState {
                &mut self.0
            }
            fn get_layer_state(&self) -> &LayerState {
                &self.0
            }
        }
    };
    gen.into()
}
//macro_rules! inherit_state {
//    ($struct_name:ident) => {
//        impl InheritState for $struct_name {
//            fn get_mut_layer_state(&mut self) -> &mut LayerState {
//                &mut self.0
//            }
//            fn get_layer_state(&self) -> &LayerState {
//                &self.0
//            }
//        }
//    };
//}
