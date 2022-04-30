use std::fs;

use super::super::{SimulationConfig, OutputNodeVectors, OutputResidualVectors, ConstData};

use ndarray::s;
use hdf5::{File, Result};


/// Creates a new blank hdf5 file to store information about the simulation
pub fn init_hdf5(config: &SimulationConfig) -> Result<()> {
    // the size of the output arrays
    let output_size = {
        if config.num_iter % config.output_frq == 0 {
            config.num_iter/config.output_frq
        }
        else {
            config.num_iter/config.output_frq + 1
        }
    };

    fs::create_dir_all("./output").unwrap();
    let file = File::create("./output/simulation.h5")?;

    // Create the simulation output group
    let sim_out = file.create_group("SimOut")?;
    sim_out.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("u")?;
    sim_out.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("rho")?;
    sim_out.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("T")?;
    sim_out.new_dataset::<f64>()
        .shape([output_size as usize])
        .create("time")?;

    // residuals of the variables
    let residuals = sim_out.create_group("Residuals")?;
    residuals.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("du")?;
    residuals.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("drho")?;
    residuals.new_dataset::<f64>()
        .shape([config.num_points as usize, output_size as usize])
        .create("dT")?;

    let sim_out_freq = sim_out.new_attr::<u32>().create("OutputFreq")?;
    sim_out_freq.write_scalar(&config.output_frq)?;
    let sim_out_nodes = sim_out.new_attr::<u32>().create("NodeNumber")?;
    sim_out_nodes.write_scalar(&config.num_points)?;

    // Create the group containing all the constant data of the simulation
    let const_data = file.create_group("ConstData")?;
    const_data.new_dataset::<f64>()
        .shape([config.num_points as usize])
        .create("x")?;
    const_data.new_dataset::<f64>()
        .shape([config.num_points as usize])
        .create("A")?;

    // Create the group containing the exact solution for this flow problem
    file.create_group("ExactSol")?;

    // Create the group were all the post-processing stuff will go
    file.create_group("PostProcess")?;

    Ok(())
}


/// Writes simulation data to the hdf5 file
#[allow(non_snake_case)]
pub fn write_to_hdf5(out_vecs: &OutputNodeVectors, out_res_vecs: &OutputResidualVectors, time: f64, i: usize) -> Result<()> {

    let file = File::open_rw("./output/simulation.h5")?;

    let slice = s![.., i];

    let u_ds = file.dataset("SimOut/u")?;
    u_ds.write_slice(&out_vecs.u, slice)?;
    let rho_ds = file.dataset("SimOut/rho")?;
    rho_ds.write_slice(&out_vecs.rho, slice)?;
    let T_ds = file.dataset("SimOut/T")?;
    T_ds.write_slice(&out_vecs.T, slice)?;

    let du_ds = file.dataset("SimOut/Residuals/du")?;
    du_ds.write_slice(&out_res_vecs.du, slice)?;
    let drho_ds = file.dataset("SimOut/Residuals/drho")?;
    drho_ds.write_slice(&out_res_vecs.drho, slice)?;
    let dT_ds = file.dataset("SimOut/Residuals/dT")?;
    dT_ds.write_slice(&out_res_vecs.dT, slice)?;

    // slice of one element
    let slice = s![i..i];

    let time_ds = file.dataset("SimOut/time")?;
    time_ds.write_slice(&[time], slice)?;

    Ok(())
}


// Writes constant data to the hdf5 file
#[allow(non_snake_case)]
pub fn write_const_hdf5(const_data: &ConstData) -> Result<()> {
    let file = File::open_rw("./output/simulation.h5")?;

    let slice = s![..];

    let x_ds = file.dataset("ConstData/x")?;
    x_ds.write_slice(&const_data.x, slice)?;
    let A_ds = file.dataset("ConstData/A")?;
    A_ds.write_slice(&const_data.A, slice)?;

    Ok(())
}
