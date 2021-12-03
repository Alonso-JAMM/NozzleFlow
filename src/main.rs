use std::f64;
use std::fs::File;
use std::io::Write;

use serde::Serialize;


struct SimulationConfig {
    num_points: u32,
    num_iter: u32,
    c: f64,
    output_frq: u32,
}


/// Stores the values of the variables at a certain point in the domain
#[allow(non_snake_case)]
#[derive(Debug, Copy, Clone)]
struct Node {
    /// velocity in the x-direction
    u: f64,
    /// temperature
    T: f64,    /// density
    rho: f64,
    /// x-position of this node
    x: f64,
    /// Area at this node
    A: f64,
    /// Determine if the corresponding variables are fixed (whose values don't
    /// change through the simulation)
    u_fixed: bool,
    T_fixed: bool,
    rho_fixed: bool,
}

impl Node {
    fn new() -> Node {
        Node {
            u: 0.0,
            T: 0.0,
            rho: 0.0,
            x: 0.0,
            A: 0.0,
            u_fixed: false,
            T_fixed: false,
            rho_fixed: false,
        }
    }
}


/// Stores the values of the derivatives with respect to time at the nodes
#[allow(non_snake_case)]
#[derive(Debug, Copy, Clone, Serialize)]
struct DeltNode {
    du: f64,
    dT: f64,
    drho: f64,
}

impl DeltNode {
    fn new() -> DeltNode {
        DeltNode {
            du: 0.0,
            dT: 0.0,
            drho: 0.0,
        }
    }
}


/// Stores the values to be saved to file
#[allow(non_snake_case)]
#[derive(Debug, Copy, Clone, Serialize)]
struct OutputValues {
    /// index of the node
    i: u32,
    /// non-dimensional x-position
    x: f64,
    /// non-dimensional area
    A: f64,
    /// non-dimensional density
    rho: f64,
    /// non-dimensional velocity
    u: f64,
    /// non-dimensional temperature
    T: f64,
}

impl OutputValues {
    fn new(node: &Node, idx: u32) -> OutputValues {
        OutputValues {
            i: idx,
            x: node.x,
            A: node.A,
            rho: 0.0,
            u: 0.0,
            T: 0.0,
        }
    }

    fn set_values(&mut self, node: &Node) {
        self.rho = node.rho;
        self.u = node.u;
        self.T = node.T;
    }
}



fn nozzle_area(x: f64) -> f64 {
    1.0 + 2.2*(x - 1.5).powi(2)
}

#[allow(non_snake_case)]
fn main() {
//     Setup the problem
    let config = SimulationConfig {
        num_points: 31,
        num_iter: 1400,
        c: 0.5,
        output_frq: 2,
    };

    let gamma = 1.4;

    let n = config.num_points as usize;

    let mut nodes = vec![Node::new(); n];
    let mut delta_nodes = vec![DeltNode::new(); n];
    // Predicted values of the nodes at new time step
    let mut predict_nodes = vec![Node::new(); n];
    let mut predict_deltas = vec![DeltNode::new(); n];

    // output variables
    let mut output_nodes: Vec<OutputValues> = Vec::new();


    // We know the boundary conditions:
    // --- node1: rho and T are fixed
    // --- nodeN: all variables are fixed
    nodes[0].T_fixed = true;
    nodes[0].rho_fixed = true;

    // The variables depend on x, and we know that x = 0 at node1 and x = 3 at nodeN
    // so we  also set the x-values here
    let delta_x = (3.0-0.0) / ((config.num_points-1) as f64);
    let mut h: f64;
    let mut new_x : f64;
    for (i, node) in nodes.iter_mut().enumerate() {
        h = i as f64;
        new_x = h * delta_x;
        node.x = new_x;
        node.A = nozzle_area(new_x);
        node.rho = 1.0 - 0.3146*new_x;
//         node.rho = 1.0 - 0.9*(new_x*10.0).tanh();
        node.T = 1.0 - 0.2314*new_x;
//         node.T = 1.0 - 0.6*(new_x*10.0).tanh();
        node.u = (0.1 + 1.09*new_x)*node.T.sqrt();
    }

    // Copy the values of nozzle areas to predicted nodes (these values are constant)
    for (predict_node, node) in predict_nodes.iter_mut().zip(nodes.iter()) {
        predict_node.A = node.A;
    }

    for (i, node) in nodes.iter().enumerate() {
        output_nodes.push(OutputValues::new(node, i as u32));
    }

    // At this point we know the values of the variables at t = 0
    // so we can continue with the  simulion!

//     Solve the problem
    let mut a_i: f64;     // speed of sound
    let c = config.c;
    let mut delta_time = vec![0.0; n];         // time steps for each
    let mut t_step;
    let mut node_im1: &Node;
    let mut node_i: &Node;
    let mut node_ip1: &Node;
    let mut delta_node_i: &mut DeltNode;
    let mut delta_u: f64;
    let mut delta_A: f64;
    let mut delta_T: f64;
    let mut delta_rho: f64;
    let mut rho_i: f64;
    let mut u_i: f64;
    let mut T_i: f64;
    let mut drho_avg: f64;
    let mut dT_avg: f64;
    let mut du_avg: f64;
    let mut total_time = 0.0;
    for i in 0..config.num_iter {
        // Calculate the time step
        for (d_t, node) in delta_time.iter_mut().zip(nodes.iter()) {
            a_i = node.T.sqrt();
            u_i = node.u;
            *d_t = c*delta_x/(a_i + u_i);
        }
        t_step = delta_time.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        total_time += t_step;

        // Calculate derivatives at this time step using forward differences
        for i in 1..(n-1) {
            node_i = &nodes[i];
            node_ip1 = &nodes[i+1];
            delta_node_i = &mut delta_nodes[i];

            delta_u = (node_ip1.u - node_i.u)/delta_x;
            delta_A = (node_ip1.A.ln() - node_i.A.ln())/delta_x;
            delta_rho = (node_ip1.rho - node_i.rho)/delta_x;
            delta_T = (node_ip1.T - node_i.T)/delta_x;

            rho_i = node_i.rho;
            u_i = node_i.u;
            T_i = node_i.T;

            delta_node_i.drho = -rho_i*delta_u - rho_i*u_i*delta_A - u_i*delta_rho;
            delta_node_i.du = -u_i*delta_u - (1.0/gamma)*(delta_T + (T_i/rho_i)*delta_rho);
            delta_node_i.dT = -u_i*delta_T - (gamma-1.0)*T_i*(delta_u + u_i*delta_A);
        }

        // Predict values of the variables at next time step
        for ((predict_node, node), delta_node) in predict_nodes.iter_mut()
                                                               .zip(nodes.iter())
                                                               .zip(delta_nodes.iter()) {

            predict_node.rho = node.rho + t_step*delta_node.drho;
            predict_node.u = node.u + t_step*delta_node.du;
            predict_node.T = node.T + t_step*delta_node.dT;
        }

        // Get the predicted derivatives at next time step from the predicted
        // values of the variables using backward differences
        for i in 1..(n-1) {
            node_i = &predict_nodes[i];
            node_im1 = &predict_nodes[i-1];
            delta_node_i = &mut predict_deltas[i];

            delta_u = (node_i.u - node_im1.u)/delta_x;
            delta_A = (node_i.A.ln() - node_im1.A.ln())/delta_x;
            delta_rho = (node_i.rho - node_im1.rho )/delta_x;
            delta_T = (node_i.T - node_im1.T)/delta_x;

            rho_i = node_i.rho;
            u_i = node_i.u;
            T_i = node_i.T;

            delta_node_i.drho = -rho_i*delta_u - rho_i*u_i*delta_A - u_i*delta_rho;
            delta_node_i.du = -u_i*delta_u - (1.0/gamma)*(delta_T + (T_i/rho_i)*delta_rho);
            delta_node_i.dT = -u_i*delta_T - (gamma-1.0)*T_i*(delta_u + u_i*delta_A);
        }

        // Get the corrected values of the variables at the new time step
        for ((node, delta_node), predict_delta) in nodes.iter_mut()
                                                        .zip(delta_nodes.iter())
                                                        .zip(predict_deltas.iter())
        {
            drho_avg = 0.5*(delta_node.drho + predict_delta.drho);
            du_avg = 0.5*(delta_node.du + predict_delta.du);
            dT_avg = 0.5*(delta_node.dT + predict_delta.dT);

            if !node.rho_fixed {
                node.rho = node.rho + t_step*drho_avg;
            }
            if !node.u_fixed {
                node.u = node.u + t_step*du_avg;
            }
            if !node.T_fixed {
                node.T = node.T + t_step*dT_avg;
            }
        }

        // Finally calculate the boundary values
        nodes[0].u = 2.0*nodes[1].u - nodes[2].u;
        nodes[n-1].u = 2.0*nodes[n-2].u - nodes[n-3].u;
        nodes[n-1].rho = 2.0*nodes[n-2].rho - nodes[n-3].rho;
        nodes[n-1].T = 2.0*nodes[n-2].T - nodes[n-3].T;

        // save variable values for post-processing
        if i % config.output_frq == 0 {
            let dir_name =  format!("./output/{}", i);
            let node_file_name = format!("./output/{}/node", i);
            let time_file_name = format!("./output/{}/time", i);
            let residual_file_name = format!("./output/{}/residual", i);
            std::fs::create_dir_all(dir_name).expect("Unable to create directory");

            // write the variables values of the nodes to file
            let mut f = File::create(node_file_name).expect("Unable to create file.");
            for (output_node, node) in output_nodes.iter_mut().zip(nodes.iter()) {
                output_node.set_values(node);
                let json_output = serde_json::to_string(output_node).unwrap();
                f.write_all(json_output.as_bytes()).expect("Unable to write to file");
                f.write_all(b"\n").expect("Unable to write to file");
            }

            // write the time to file
            let mut f = File::create(time_file_name).expect("Unable to create file.");
            f.write_all(total_time.to_string().as_bytes()).expect("Unable to write to file");

            // write residual values to file
            let mut residual = DeltNode::new();
            let mut f = File::create(residual_file_name).expect("Unable to create file.");
            for (delta_node, predict_delta) in delta_nodes.iter().zip(predict_deltas.iter()) {
                residual.drho = 0.5*(delta_node.drho + predict_delta.drho);
                residual.du = 0.5*(delta_node.du + predict_delta.du);
                residual.dT = 0.5*(delta_node.dT + predict_delta.dT);

                let json_output = serde_json::to_string(&residual).unwrap();
                f.write_all(json_output.as_bytes()).expect("Unable to write to file");
                f.write_all(b"\n").expect("Unable to write to file");

            }
        }
    }
//     println!("{}", total_time);

//     for (i, d_node) in delta_nodes.iter().enumerate() {
//         println!("{}: {:?}", i, d_node.dT);
//     }

//     for (i, node) in nodes.iter().enumerate() {
//         println!("{}: {:?}", i, node.rho);
//     }

//     for (i, node) in predict_nodes.iter().enumerate() {
//         println!("{}: {:?}", i, node.T);
//     }

//     for (i, d_node) in predict_deltas.iter().enumerate() {
//         println!("{}: {:?}", i, d_node.dT);
//     }



//     Output solution
}
