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
////also impl the following
//impl InitializeLayer for T_layer{
//    fn init() -> Self {
//        T_layer(LayerState {
//            input: None,
//            input_size: 0,
//            output_size: 0,
//            width: 0,
//            activation: None,
//            dtype: DataType::Float,
//        })
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
    //allow non inheritence (if self.0 is self.layer_state)
    let is_inherited = match ast.data {
        syn::Data::Struct(ref data) => match data.fields {
            syn::Fields::Named(ref fields) => fields
                .named
                .iter()
                .any(|field| field.ident.as_ref().unwrap() == "0"),
            _ => true,
        },
        _ => false,
    };
    if is_inherited {
        let gen = quote! {
            impl InheritState for #name{
                fn get_mut_layer_state(&mut self) -> &mut LayerState {
                    &mut self.0
                }
                fn get_layer_state(&self) -> &LayerState {
                    &self.0
                }
            }
            impl InitializeLayer for #name {
                fn init() -> Self {
                    #name(LayerState {
                        input: None,
                        input_size: 0,
                        output_size: 0,
                        width: 0,
                        activation: None,
                        dtype: DataType::Float,
                    })
                }
            }
        };
        gen.into()
    } else {
        //get all fields except self.layer_state in a list
        let fields = match ast.data {
            syn::Data::Struct(ref data) => match data.fields {
                syn::Fields::Named(ref fields) => fields
                    .named
                    .iter()
                    .filter(|field| field.ident.as_ref().unwrap() != "layer_state")
                    .map(|field| field.ident.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                _ => vec![],
            },
            _ => vec![],
        };
        let gen = quote! {
            impl InheritState for #name{
                fn get_mut_layer_state(&mut self) -> &mut LayerState {
                    &mut self.layer_state
                }
                fn get_layer_state(&self) -> &LayerState {
                    &self.layer_state
                }
            }
            impl InitializeLayer for #name {
                fn init() -> Self {
                    #name {
                        layer_state: LayerState {
                            input: None,
                            input_size: 0,
                            output_size: 0,
                            width: 0,
                            activation: None,
                            dtype: DataType::Float,
                        }
                        //also initialize any other fields as long as they have
                        //default implementation
                        #(, #fields: Default::default())*
                    }
                }
            }
        };
        gen.into()
        //let gen = quote! {
        //        impl InheritState for #name {
        //            fn get_mut_layer_state(&mut self) -> &mut LayerState {
        //                &mut self.0
        //            }
        //            fn get_layer_state(&self) -> &LayerState {
        //                &self.0
        //            }
        //}
    }
}
