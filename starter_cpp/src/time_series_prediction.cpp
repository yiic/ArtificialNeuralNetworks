// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include <iomanip>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "preprocess.h"
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <cmath>
#include <math.h>
#include <algorithm>


#ifdef WINDOWS
#    include <windows.h>
#else
#    include <sys/time.h>
#endif

double seconds()
{
#ifdef WINDOWS
    return (double)GetTickCount() * 1e-3;
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
#endif
}

using namespace std;
const double pi = 3.14159265358979323846;


class Layer
{
public: Vec activation;
public: Vec blame;
public: int weights_dim;
    
public:
    Layer(size_t inputs, size_t outputs) :
    activation(outputs),
    blame(outputs){
        weights_dim = outputs + inputs * outputs;
    }
    virtual ~Layer(){
    }
    
    // Basic functions
    virtual void activate( Vec& weights, const Vec& x) = 0;
    virtual void backprop( Vec& weights, Vec& prevBlame) = 0;
    virtual void update_gradient(const Vec& x, Vec& gradient) = 0;
};

//=======================================================================
// LayerLinear
//=======================================================================
class LayLinear : public Layer
{
public:
    LayLinear (size_t inputs, size_t outputs) :
    Layer(inputs,outputs){
    }
    
    virtual ~LayLinear(){
    }
    
    // ---------------------------------------------------
    // Activate
    virtual void activate( Vec& weights, const Vec& x){
        
        VecWrapper b(weights, 0, weights.size() / (1 + x.size()));
        VecWrapper M(weights, b.size(), weights.size());
        
        for(size_t i = 0; i < b.size(); i++){
            VecWrapper m(M, x.size()*i, x.size());
            activation[i] = m.dotProduct(x);
        }
        
        activation += b;
    }
    
    // ---------------------------------------------------
    // Backpropagation -> prevBlame
    virtual void backprop( Vec& weights, Vec& prevBlame){
        
        int output_size = blame.size();
        int input_size = prevBlame.size();
        
        VecWrapper b(weights, 0, blame.size());
        VecWrapper M(weights, b.size(), input_size * output_size);
        
        // prevBlame = M^T * blame
        Vec mt_row(output_size);
        for(int i = 0; i < input_size; i++){
            for(int j = 0; j < output_size; j++)
                mt_row[j] = M[j * input_size + i];
            
            prevBlame[i] = mt_row.dotProduct(blame);
        }
        
    }
    
    // ---------------------------------------------------
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        VecWrapper gb(gradient, 0, blame.size());
        VecWrapper gM(gradient, gb.size(), gradient.size());
        
        // update b
        gb += blame;

        // update M
        Matrix temp(outer_product(blame, x));

        
        for(size_t i = 0; i < temp.rows(); i++){
            for (size_t j = 0; j < temp.cols(); j++){
                gM[i * temp.cols() + j] += temp[i][j];
                
            }
        }
        
    }
    
    // ---------------------------------------------------
    // Ordinary_least_squares: compute M and b by knowing y and x
    void ordinary_least_squares( const Matrix& X,  const Matrix& Y, Vec& weights){
        
        Vec x_centroid(X.cols());
        Vec y_centroid(Y.cols());
        
        Matrix origin_centered_x(0,X.cols());
        Matrix origin_centered_y(0,Y.cols());
        
        for(size_t i = 0; i < X.cols(); i++)
            x_centroid.data()[i] = X.columnMean(i);
        for(size_t i = 0; i < Y.cols(); i++)
            y_centroid.data()[i] = Y.columnMean(i);
        
        // origin centered data
        for(size_t i = 0; i < X.rows(); i++){
            origin_centered_x.newRow().copy(X.crow(i) - x_centroid);
        }
        
        for(size_t i = 0; i < Y.rows(); i++){
            origin_centered_y.newRow().copy(Y.crow(i) - y_centroid);
        }
        
        // calculate M
        Matrix yx(origin_centered_y.cols(), origin_centered_x.cols());
        yx.fill(0.0);
        Matrix xx(origin_centered_x.cols(), origin_centered_x.cols());
        xx.fill(0.0);
        
        for(size_t i = 0; i < origin_centered_y.rows(); i++){
            Matrix temp_yx(outer_product(origin_centered_x.row(i), origin_centered_y.row(i)));
            for(size_t k = 0; k < origin_centered_y.cols(); k++)
                yx.row(k) += temp_yx.row(k);
        }
        
        for(size_t i = 0; i < origin_centered_x.rows(); i++){
            Matrix temp_xx = outer_product(origin_centered_x.row(i), origin_centered_x.row(i));
            for(size_t k = 0; k < origin_centered_x.cols(); k++)
                xx.row(k) += temp_xx.row(k);
        }
        
        Matrix *xx_inverse = xx.pseudoInverse();
        Matrix *M = xx_inverse->multiply(yx, *xx_inverse,false,false);
        
        // calculate b
        Vec b(M->rows());
        for(size_t i = 0; i < M->rows(); i++){
            b.data()[i]= M->row(i).dotProduct(x_centroid);
        }
        b.copy(y_centroid  - b);
        
        //add b to weights
        weights.resize(b.size() + M->rows()*M->cols());
        for(size_t i = 0; i < b.size(); i++){
            weights.data()[i] = b.data()[i];
        }
        
        //add M to weights
        for(size_t i = 0; i < M->rows(); i++){
            for(size_t j = 0; j < M->cols(); j++){
                weights.data()[b.size()+i*M->cols()+j] = M->row(i).data()[j];
            }
        }
        
    }
    
    // ---------------------------------------------------
    // Outer Product of two Vec's : a * b
    Matrix outer_product(Vec& a, Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }

    Matrix outer_product(const Vec& a,  Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }
    
    Matrix outer_product(Vec& a, const Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }
    
    
};

//=======================================================================
// LayerTanh
//=======================================================================
class LayerTanh : public Layer
{
public:
    LayerTanh (size_t inputs) :
    Layer(inputs,inputs){
        weights_dim = 0;
    }
    
    virtual ~LayerTanh(){
    }
    // ---------------------------------------------------
    // Activate
    virtual void activate(Vec& weights, const Vec& x){
        for(size_t i = 0; i < x.size(); i++){
            activation[i] = tanh(x[i]);
        }
    }
    // ---------------------------------------------------
    // Backpropagation
    virtual void backprop( Vec& weights, Vec& prevBlame){
        for(size_t i = 0; i < prevBlame.size(); i++){
            prevBlame[i] = blame[i] * (1.0 - (activation[i] * activation[i]));
        }
    }
    
    // ---------------------------------------------------
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        
    }
    
};

//=======================================================================
// LayerSinusoid
//=======================================================================
class LayerSinusoid : public Layer
{
public: Vec layerinput;
public:
    LayerSinusoid (size_t inputs) :
    Layer(inputs,inputs){
        weights_dim = 0;
    }
    
    virtual ~LayerSinusoid(){
    }
    // ---------------------------------------------------
    // Activate
    virtual void activate(Vec& weights, const Vec& x){
        for(size_t i = 0; i < x.size() - 1; i++){
            activation[i] = sin(x[i]);
        }
        activation[x.size() - 1] = x[x.size() - 1];
        
        layerinput.copy(x);
    }
    // ---------------------------------------------------
    // Backpropagation
    virtual void backprop( Vec& weights, Vec& prevBlame){
        
        for(size_t i = 0; i < prevBlame.size() - 1; i++){
            prevBlame[i] = blame[i] * cos(layerinput[i]);
        }
        prevBlame[prevBlame.size() - 1] = blame[prevBlame.size() - 1] ;
        
    }
    
    // ---------------------------------------------------
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        
    }
    
};


//=======================================================================
// NeuralNet
//=======================================================================
class NeuralNet : public SupervisedLearner
{
public:
    Vec weights;
    Vec predi;
    
    vector<int> net_info;
    vector<Layer*> layers;
    
    // Constructors
    NeuralNet(vector<int> net_sets): weights(1){
        net_info = net_sets;
        
        // set layers
        layers.push_back(new LayLinear(net_sets[0], net_sets[1]));
        layers.push_back(new LayerSinusoid(net_sets[1]));
        layers.push_back(new LayLinear(net_sets[2], net_sets[3]));
        
        int weights_len = net_sets[1] + net_sets[1] * net_sets[0] + net_sets[3] + net_sets[2] * net_sets[3];
        
        weights.resize(weights_len);
    }
    
    NeuralNet(size_t w): weights(w){ }
    NeuralNet(){ }
    
    virtual ~NeuralNet(){
        for(size_t i = 0; i < layers.size(); i++)
            delete(layers[i]);
    }
    virtual const char* name() { return "NeuralNet"; }
    virtual void train(const Matrix& features, const Matrix& labels){ }
    
    //----------------------------------------------------------
    // Initial Weights
    virtual void initWeights(){
        
        int pos = 0;
        Rand rand(2);
        
        for(int i = 0; i < layers.size(); i+=2){
            VecWrapper layer_weights(weights, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            // initialize first linear layer
            if(i == 0){
                /// bias (output)
                for(int j = 0; j < 100; j++){
                    if(j < 50){
                        layer_weights[j] = pi;
                    }else{
                        layer_weights[j] = pi / 2.0;
                    }
                }
                layer_weights[100] = 0.0;
            
                /// weights (input * output)
                for(int j = 101; j < layer_weights.size() - 1; j++){
                    if (j % 50 != 0)
                        layer_weights[j] = (j % 50) * 2.0 * pi;
                    else
                        layer_weights[j] = 50.0 * 2.0 * pi;
                }
                layer_weights[layer_weights.size() - 1] = 0.01;
            }
            
            // initialize last linear layer
            else{
                
                /// bias
                layer_weights[0] = 0.0;
                
                /// weights
                for(int j = 1; j < layer_weights.size(); j++){
                    layer_weights[j] = 0.01;
                }
                
            }
            
        }

    }
    
    
    //----------------------------------------------------------
    // Predict -> activation
    virtual const Vec& predict(const Vec& net_input){
        int pos = 0;
        
        // activate all layers
        for(int i = 0; i < layers.size(); i++){
            VecWrapper layer_weights(weights, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            if(i == 0){
                layers[i]->activate(layer_weights, net_input);
            }else{
                layers[i]->activate(layer_weights, layers[i - 1]->activation);
            }
        }
    
        layers[layers.size() - 1]->activation[0] = layers[layers.size() - 1]->activation[0];
        predi.copy(layers[layers.size() - 1]->activation);
        
        return predi;
    }
    
    //----------------------------------------------------------
    // Backprop
    virtual void backProp(const Vec targets){
        
        // output layer blame
        layers[layers.size() - 1]->blame.copy(targets - layers[layers.size() - 1]->activation);
       
        
        // back propagation
        int pos = 0;
        for(int i = layers.size() - 1; i > 0; i--){
            pos += layers[i]->weights_dim;
            
            VecWrapper layer_weights(weights, weights.size() - pos, layers[i]->weights_dim);
            
            layers[i]->backprop(layer_weights, layers[i - 1]->blame);
        }
    }
    
    //----------------------------------------------------------
    // Update Gradient
    virtual void updateGradient(const Vec& net_input, Vec& gradients){
        int pos = 0;
    
        for(int i = 0; i < layers.size(); i++){
            VecWrapper layer_gradients(gradients, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            if (i == 0)
                layers[i]->update_gradient(net_input, layer_gradients);
            else
                layers[i]->update_gradient(layers[i - 1]->activation, layer_gradients);
            
        }
        
    }
    
    //----------------------------------------------------------
    // Refine Weights (gradient decent) --> Stochastic
    virtual void refineWeights(const Vec& net_input, const Vec& targets, double lr, double lambda){
        Vec gradients(weights.size());
       
        predict(net_input);
        backProp(targets);
        gradients.fill(0.0);
        updateGradient(net_input, gradients);
        
        
        //Update weights
        
        
        //============================================
//        // No regulation
//        weights += gradients * lr;
        
        //============================================
//        // L2 regulation
//        VecWrapper last_layer_ws(weights, 202, 101);
//        last_layer_ws *= (1 - lambda);
        
//        weights += gradients * lr;

        
        //============================================
        // L1 regulation
        VecWrapper last_layer_ws(weights, 202, 101);

        for(int i = 0; i < last_layer_ws.size(); i++){
            if(last_layer_ws[i] > 0)
                last_layer_ws[i] - = lambda;
            else
                last_layer_ws[i] += lambda;
        }

        weights += gradients * lr;
        
        
        
    }
    

}; // end of NeuralNet

//=======================================================================
// Filter (Data Prepocess)
//=======================================================================
class Filter{
    
public: NomCat nomcat;
public: Normalizer normal;
public: Imputer inputer;
public: Matrix after;
    
public:
    Filter(Matrix& before, Matrix& after){
        Matrix middle1, middle2;
        
        //Imputer
        inputer.train(before);
        middle1.setSize(before.rows(), inputer.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
            inputer.transform(before[i], middle1[i]);
        }
    
        //Nomcat
        nomcat.train(before);
        middle2.setSize(before.rows(), nomcat.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
                nomcat.transform(middle1[i], middle2[i]);
        }
        
        //Normalizer
        normal.train(middle2);
        after.setSize(before.rows(), normal.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
                normal.transform(middle2[i], after[i]);
        }
        //cout << "Data prepocess is done." <<endl;
    }

    virtual ~Filter(){}
    
    // Untransform a Vec of pattern data
    Vec untran(Vec in){
        Vec mid;
        Vec out;
        normal.untransform(in, mid);
        nomcat.untransform(mid, out);
        return out;
    }

};

//=======================================================================
// "debug spew" test
//=======================================================================
void h5_debug_test(){
    vector<int> net_sets = {1, 101, 101, 1};
    Vec fea1({0});
    Vec tar1({3.4});
    
    Vec fea2({0.00390625});
    Vec tar2({3.8});
    
    Vec fea3({0.0078125});
    Vec tar3({4.0});

    NeuralNet test_net(net_sets);
    test_net.initWeights();
    
    test_net.refineWeights(fea1, tar1, 0.001, 1);
    test_net.refineWeights(fea2, tar2, 0.001, 1);
    
}

//=======================================================================
// Labor Statistics test
//=======================================================================
void Labor_Stat(){
    
    //Load the data
    string fn = "../bin/data/";
    Matrix labels_raw;
    labels_raw.loadARFF(fn + "labor_stats.arff");
    
    // train data
    Matrix train_features(256, 1);
    Matrix train_labels(256, 1);
    
    for(int i = 0; i < train_features.rows(); i++)
        train_features[i][0] = i / 256.00;
    
    for(int i = 0; i < train_labels.rows(); i++)
        train_labels[i][0] = labels_raw[i][0];
    
 
    // test data
    Matrix test_features(357, 1);
    Matrix test_labels(357, 1);
    
    for(int i = 0; i < test_features.rows(); i++)
        test_features[i][0] = i / 256.00;
    
    for(int i = 0; i < test_labels.rows(); i++)
        test_labels[i][0] = labels_raw[i][0];
    
    
    // Initailize the NeuralNet
    vector<int> net_sets = {1, 101, 101, 1};
    NeuralNet bp_learner(net_sets);
    bp_learner.initWeights();
    
    
    // TRAIN
    double lr = 0.01;
    
//    //L2
//    double lambda = 0.00001;
    
    //L1
    double lambda = 0.000004;
    
    int epoch = 1000;
    
    for(int e = 0; e < epoch; e++){
        if(train_features.rows() != train_labels.rows())
            throw Ex("Mismatching number of rows");
        else{
            // create an vector for row index
            vector<int> select_index(train_features.rows());
            int temp = 0;
            for(int i = 0; i < select_index.size(); i++)
                select_index[i] = i;
            
            // loop over every pattern randomly
            random_shuffle(select_index.begin(), select_index.end());
            for(vector<int>::iterator it = select_index.begin(); it != select_index.end(); ++it){
                
                // update weights
                bp_learner.refineWeights(train_features[*it], train_labels[*it], lr, lambda);
                
            }//end shuffle for
            
        }//end else
        
    }//end epoch training for
    
    
    // TEST
    Vec predict; double prediction;
    double target;
    for(int i = 0; i < test_features.rows(); i++){
        
        predict.copy(bp_learner.predict(test_features[i]));
        prediction = predict[0];
        cout << test_features[i][0] << ", " << prediction << endl;
        
    }
    
}


//=======================================================================
int main(int argc, char *argv[])
{
    enableFloatingPointExceptions();
    
    int ret = 1;
    try{
        
        //h5_debug_test();
   
        Labor_Stat();
        
        
        
        ret =0;
    }catch(const std::exception& e){
        cerr << "An error occurred: " << e.what() << "\n";
    }
    
    
    return ret;
}

