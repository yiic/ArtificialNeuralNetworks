// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <fstream>
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
    // Backpropagation -> prevBlame
    virtual void backprop_toinput( Vec& weights, Vec& prevBlame){
        
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
        int weights_len = 0;
        
        for (int i = 0; i < net_sets.size() - 1; i++){
            if(i % 2 == 0)
            {
                layers.push_back(new LayLinear(net_sets[i], net_sets[i + 1]));
                weights_len += net_sets[i + 1] + net_sets[i + 1] * net_sets[i];
            }
            else
            {
                layers.push_back(new LayerTanh(net_sets[i + 1]));
            }
        }
        
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
        Rand rand(2);
        int pos = 0;
        int input_dim;
        
        for(int i = 0; i < layers.size(); i+=2){
            VecWrapper layer_weights(weights, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            input_dim = layers[i]->weights_dim / layers[i]->blame.size() - 1;
            
            for(int j = 0; j < layer_weights.size(); j++){
                layer_weights[j] = max(0.03, 1.0 / input_dim) * rand.normal();
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
    // Backprop to input layer
    virtual void backProp_toinput(const Vec targets, Vec& input_blame){

        // output layer blame
        layers[layers.size() - 1]->blame.copy(targets - layers[layers.size() - 1]->activation);

        // back propagation
        int pos = 0;
        for(int i = layers.size() - 1; i >= 0; i--){
            pos += layers[i]->weights_dim;
            VecWrapper layer_weights(weights, weights.size() - pos, layers[i]->weights_dim);
            if (i != 0)
            {
                layers[i]->backprop(layer_weights, layers[i - 1]->blame);
            }else
            {
                layers[i]->backprop(layer_weights, input_blame);
            }
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
    // Refine Weights
    virtual void refineWeights(const Vec& net_input, const Vec& targets, double lr){
        Vec gradients(weights.size());
       
        // ( Stochastic )
        predict(net_input);// forwardProp
        backProp(targets);
        gradients.fill(0.0);
        updateGradient(net_input, gradients);

        weights += gradients * lr;
        
    }
    
    //----------------------------------------------------------
    // Refine Weights and Inputs
    virtual void refineWeightsNInput(Vec& net_input, Vec& v, const Vec& targets, double lr){
        
        // refine weights
        Vec gradients(weights.size());
        gradients.fill(0.0);
        
        Vec input_blame(net_input.size());
        
        predict(net_input);// forwardProp
        backProp_toinput(targets, input_blame);
        updateGradient(net_input, gradients);
        
        weights += gradients * lr;
        
        //refine inputs V[t] = v
        v[0] += input_blame[2] * lr;
        v[1] += input_blame[3] * lr;
        
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
// Observation Function
//=======================================================================
void train_with_images(const Matrix& X, Matrix& V, NeuralNet& unsup_learner){

    int width = 64;
    int height = 48;
    int channels = X.cols() / (width * height);

    Rand rand(2);
    double lr = 0.1;

    for(int j = 0; j < 10; j++){
        for(int i = 0; i < 10000000; i++){

            int t = rand.next(X.rows());
            int p = rand.next(width);
            int q = rand.next(height);
            int s = channels * (width * q + p);

            Vec features({p/(width * 1.0), q/(height * 1.0), V[t][0], V[t][1]});
            Vec labels(channels);
            for (int h = 0; h < channels; h++)
                labels[h] = X[t][s + h];

            unsup_learner.refineWeightsNInput(features, V[t], labels, lr);

            if (i % 500000 == 0)
                cout << V[t][0] << ", " << V[t][1] << " regine weights and inputs..." << endl;

        }
        lr *= 0.75;
        cout << "train with images: lr decay... " << j << endl;
    }

    // Plot the intrinsic data V when trainig is done
    for (int h = 0; h < V.rows(); h++)
        cout << V[h][0] << ", " << V[h][1] << endl;
}
    
    
//=======================================================================
// Transation Function
//=======================================================================
void train_with_actions(const Matrix& actions, const Matrix& V, NeuralNet& sup_learner){

    Rand rand(2);
    double lr = 0.1;
    
    // train states matrix (has one fewer row than V)
    Matrix V_train(V.rows() - 1, V.cols());
    V_train.copyBlock(0, 0, V, 0, 0, V.rows() - 1, V.cols());
    
    for(int j = 0; j < 10; j++){
        for(int i = 0; i < 100000; i++){
            
            int t = rand.next(V_train.rows());
            
            Vec features({V_train[t][0], V_train[t][1], actions[t][0], actions[t][1], actions[t][2], actions[t][3]});
            Vec labels({V[t + 1][0] - V_train[t][0], V[t + 1][1] - V_train[t][1]}); // difference as label
            
            sup_learner.refineWeights(features, labels, lr);
            
            if (i % 5000 == 0)
                cout << "training... "; Vec pred = sup_learner.predict(features); pred.print(); cout << endl;
            
        }
        lr *= 0.75;
        cout << "train with actions: lr decay... " << j << endl;
    }
    
   
}

//=======================================================================
// Draw SVG images
//=======================================================================
unsigned int rgbToUint(int r, int g, int b){
    return (0xff000000 | ((r & 0xff) << 16)) | ((g & 0xff) << 8) | ((b & 0xff));
}

void makeImage(Vec& state, NeuralNet& observ_learner, const string filename){
    Vec in;
    in.resize(4);
    in[2] = state[0];
    in[3] = state[1];
    
    Vec out;
    //Image im;
    int w = 64;
    int h = 48;
    GSVG svg(w, h, 0, 0, 0, 0, 0);
    
    //im.resize(w, h);
    
    for(int y = 0; y < h; y++){
        in[1] = (double)y / h;
        for(size_t x = 0; x < w; x++){
            in[0] = (double)x / w;
            
            out.copy(observ_learner.predict(in));
            
            unsigned int color = rgbToUint(out[0] * 256, out[1] * 256, out[2] * 256);
            svg.rect(x, y, 1, 1, color);
           
        }
    }
    
    std::ofstream s;
    s.exceptions(std::ios::badbit);
    s.open(filename +".svg", std::ios::binary);
    svg.print(s);
}

// test draw image
void test_draw_image(){
   GSVG svg(64,48, 0, 0, 0, 0, 0);
    
    for(int y = 0; y < 48; y++){
        for(size_t x = 0; x < 64; x++){
            unsigned int color = 0x008080;
            svg.rect(x, y, 1, 1, color);
            
        }
    }
    
    std::ofstream s;
    s.exceptions(std::ios::badbit);
    s.open("testimage.svg", std::ios::binary);
    svg.print(s);
    cout << "end" << endl;
}





//=======================================================================
// Train Models and Test
//=======================================================================
void train_and_test(){
    
    //----------------
    // Train
    //----------------
    Matrix X;
    Matrix act;
    string fn = "../bin/data/";
    X.loadARFF(fn + "observations.arff");
    act.loadARFF(fn + "actions.arff");
    
    // Normalize Observation
    for (int i = 0; i < X.rows(); i++)
        X[i].copy(X[i] * (1.0 / 256.0));
    
    // Initialize States matrix
    Matrix V(X.rows(), 2);
    V.fill(0.0);
    
    //train observation function
    NeuralNet observ_learner({4, 12, 12, 12, 12, 3, 3});
    observ_learner.initWeights();
    train_with_images(X, V, observ_learner);
    
    //Normalize V and actions
    Matrix actions_filted;
    Matrix V_filted;
    Filter fil_actions(act, actions_filted);
    Filter fil_V(V, V_filted);
    
    //train transition funtion
    NeuralNet trans_learner({6, 6, 6, 2});
    trans_learner.initWeights();
    train_with_actions(actions_filted, V_filted, trans_learner);
    
    
    //----------------
    // Test (generate images)
    //----------------
    Matrix controls_raw;
    controls_raw.loadARFF(fn + "controls.arff");

    Matrix controls;
    Filter fil_controls(controls_raw, controls);

    string imagename = "frame";
    
    Vec state({V_filted[0][0], V_filted[0][1]});
    Vec trans_in;
    Vec state_diff;
    
    for (int i = 0; i < controls.rows(); i++){
        //draw current state image
        Vec state_unfilted = fil_V.untran(state);
        makeImage(state_unfilted, observ_learner, imagename + to_string(i));
        
        //predict next state
        trans_in.copy({state[0], state[1], controls[i][0], controls[i][1], controls[i][2], controls[i][3]});
        state_diff.copy(trans_learner.predict(trans_in));
        
        state += state_diff;
        
        if (i == controls.rows() - 1){
            makeImage(state_unfilted, observ_learner, imagename + to_string(i+1));
        }

    }
    
    
}


//=======================================================================
int main(int argc, char *argv[])
{
    enableFloatingPointExceptions();
    
    int ret = 1;
    try{
        
        train_and_test();
        //test_draw_image();
        
        
        ret =0;
    }catch(const std::exception& e){
        cerr << "An error occurred: " << e.what() << "\n";
    }
    
    
    return ret;
}

