/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pro1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import org.la4j.LinearAlgebra;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;

/**
 *
 * @author daisy
 */
public class mahadist {

    int num_tr;
    int num_te;
    int dimension_tr, NB_tr, NC_tr; // number of training data in class A, B and C
    int dimension_te, NB_te, NC_te; // number of testing data in class A, B and C
    int numTR;

    // the position of the centroid
    double[] centroid;
    double[] mahalanobis_distance;

    // the training_data, Covariance matrix, and the X_z & X_z_T that used to calculate Covariance matrix
    double[][] training_data;
    double[][] covariance_matrix;
    double[][] training_data_T;
    double[][] X_z;
    double[][] X_z_T;
    // Matrix version and temps
    Matrix covariance_matrix_M;
    Matrix testing_data;
    Vector[] x_minus_y;
    Vector[] x_minus_y_T;

    // the vector BA, CA and CB
    double[] wab;
    double[] wac;
    double[] wbc;

    // tab is the value to be compared with wab*(instance vector) 
    // to decide whether the instance should be classified into class A/C or class B/C
    // the same as tac and tbc
    double tab;
    double tac;
    double tbc;

    public mahadist() {
        // initialize all global variables here
        num_tr = num_te = 1;
        dimension_tr = NB_tr = NC_tr = 0;
        dimension_te = NB_te = NC_te = 0;
        tab = tbc = tac = 0.0;
        wab = wbc = wac = null;
    }

    public void run(String training_file, String testing_file) {
        /**
         * 1. read training data and calculate the centroid of A, B, C
         */
        readTrainingData(training_file);

        /**
         * 2. read testing data, test it with the function and keep track.
         */
        readTestingData(testing_file);

        /**
         * 3. calculate STATISTICS and output
         */
        output();

    }

    private void readTrainingData(String file) {

        try {

            BufferedReader br = new BufferedReader(new FileReader(file));
            // read the first line to get the Dimension, number of A, B and C
            String line = br.readLine();
            line = line.trim();

            // split function is used to split the content of line by space
            String[] splited = line.split("\\s+"); //this only splited one line

            //Set the matrix M and N.
            num_tr = Integer.parseInt(splited[0]);
            dimension_tr = Integer.parseInt(splited[1]);
            centroid = new double[dimension_tr];
            training_data = X_z = new double[dimension_tr][num_tr];
            training_data_T = X_z_T = new double[num_tr][dimension_tr];
            covariance_matrix = new double[dimension_tr][dimension_tr];
            covariance_matrix_M = new Basic2DMatrix(new double[dimension_tr][dimension_tr]);

            // Optional: initialized all the Matrix (actually double arrays in this code) to be 0.0
            // Optional: could be modified for better performance
            for (int i = 0; i < dimension_tr; i++) {
                centroid[i] = 0.0;
            }
            for (int i = 0; i < dimension_tr; i++) {
                for (int j = 0; j < num_tr; j++) {
                    training_data[i][j] = X_z[i][j] = 0.0;
                }
            }
            for (int i = 0; i < num_tr; i++) {
                for (int j = 0; j < dimension_tr; j++) {
                    training_data_T[i][j] = X_z_T[i][j] = 0.0;
                }
            }
            for (int i = 0; i < dimension_tr; i++) {
                for (int j = 0; j < dimension_tr; j++) {
                    covariance_matrix[i][j] = 0.0;
                }
            }

            // In order to compute the centroid of A, 
            // we read all training data of A, add them together,  
            // and divide the sum by the total number of data in A later.
            for (int i = 0; i < num_tr; i++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                }
                splited = line.split("\\s+");
                for (int j = 0; j < dimension_tr; j++) {
                    training_data_T[i][j] = Double.parseDouble(splited[j]);
                    centroid[j] += Double.parseDouble(splited[j]);
                }
            }

            for (int i = 0; i < dimension_tr; i++) {
                centroid[i] /= num_tr;
            }

            // set training_data and X_z
            for (int i = 0; i < dimension_tr; i++) {
                for (int j = 0; j < num_tr; j++) {
                    training_data[i][j] = training_data_T[j][i];
                    X_z[i][j] -= centroid[i];
                }
            }

            // set X_z_T
            for (int i = 0; i < num_tr; i++) {
                for (int j = 0; j < dimension_tr; j++) {
                    X_z_T[i][j] = X_z[j][i];
                }
            }

            // set Covariance Matrix
            double temp = 0.0;
            for (int i = 0; i < dimension_tr; i++) {
                for (int j = 0; j < dimension_tr; j++) {
                    temp = 0.0;
                    // used for accessing the multiplier and multiplicand
                    for (int y = 0; y < num_tr; y++) {
                        temp += X_z[i][y] * X_z_T[y][j];
                    }
                    covariance_matrix_M.set(i, j, temp / 100);
                }
            }
        } catch (IOException e) {
        }

    }

    private void readTestingData(String file) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            // read the first line to get the Dimension, number of A, B and C
            String line = br.readLine();
            line = line.trim();

            // split function is used to split the content of line by space
            String[] splited = line.split("\\s+");

            num_te = Integer.parseInt(splited[0]);
            dimension_te = Integer.parseInt(splited[1]);
            mahalanobis_distance = new double[num_te];
            testing_data = new Basic2DMatrix(new double[num_te][dimension_te]);
            x_minus_y = x_minus_y_T = new Vector[num_te];
            Matrix cov_inv = new Basic2DMatrix(new double[dimension_te][dimension_te]); //covariance matrix's inverse
            double coordinate = 0;
            for (int i = 0; i < num_te; i++) {
                x_minus_y[i] = x_minus_y_T[i] = new BasicVector(new double[dimension_te]);
            }

            // initialize 
            for (int i = 0; i < num_te; i++) {
                mahalanobis_distance[i] = 0.0;
            }

            // if testing data has different dimensions from training data,
            // EXIT.
            if (dimension_tr != dimension_te) {
                System.out.println("ERROR: Different dimensions in training data and testing data!");
                System.exit(0);
            }

            // go through instance one by one
            for (int i = 0; i < num_te; i++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                }
                splited = line.split("\\s+");
                for (int j = 0; j < dimension_te; j++) {
                    coordinate = Double.parseDouble(splited[j]);
                    testing_data.set(i, j, coordinate);
                    // set x-y ( the distance between the point and the centroid)
                    x_minus_y[i].set(j, coordinate - centroid[j]);
                }
            }

            // set mahalanobis_distance. ATTENTION: NEED TO IMPORT MatrixInverter for this
            Vector product;
            double product_double;
            cov_inv = (covariance_matrix_M.withInverter(LinearAlgebra.GAUSS_JORDAN)).inverse(LinearAlgebra.DENSE_FACTORY);
            for (int i = 0; i < num_te; i++) {
                product = (x_minus_y[i].multiply(cov_inv)).multiply(x_minus_y[i].toColumnMatrix());
                product_double = product.sum();
                mahalanobis_distance[i] = Math.sqrt(product_double);
            }
        } catch (IOException e) {
        } catch (NumberFormatException e) {
        }
    }

    private void output() {
        System.out.println("Centroid:  " + Arrays.toString(centroid));

        // print matrix
        System.out.print("Covariance Matrix: \n");
        for (int i = 0; i < dimension_tr; i++) {
            System.out.print("[");
            for (int j = 0; j < dimension_tr; j++) {
                String temp = covariance_matrix_M.get(i, j) + "";
                System.out.print(" " + temp);
            }
            System.out.print("]");
            System.out.println("");
        }

        // print mahalanobis_distance
        System.out.println("Distance: ");
        for (int i = 0; i < num_te; i++) {
            System.out.print(i + 1 + ". ");
            System.out.print(testing_data.getRow(i).toString());
            System.out.print(" -- " + mahalanobis_distance[i] + "\n");
        }

    }

   
    public static void main(String args[]) {

        mahadist c = new mahadist();
        //c.run(args[0], args[1]);

        c.readTrainingData(args[0]);
        c.readTestingData(args[1]);
        c.output();
    }
}
