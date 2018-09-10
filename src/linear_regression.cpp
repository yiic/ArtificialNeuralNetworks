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
#include <cmath>
#include <algorithm>
using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;
using namespace std;


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

//============================================================
// LayerLinear
//============================================================
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

            // prevBlame = M^T * blame  Question?
            Vec mt_row(output_size);
            for(int i = 0; i < input_size; i++){
                for(int j = 0; j < output_size; j++)
                    mt_row[j] = M[j * input_size + i];

                 prevBlame[i] = mt_row.dotProduct(blame);
            }
            
        }
    
        // Update Gradient
        virtual void update_gradient( const Vec& x, Vec& gradient){
            VecWrapper gb(gradient, 0, blame.size());
            VecWrapper gM(gradient, gb.size(), gradient.size());
            
            // update b
            gb += blame;
           
            // update M
            Matrix temp(outer_product(x, blame));
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
            
        }//end of ordinary_least_squares
    
        // ---------------------------------------------------
        // Outer Product of two Vec's
        Matrix outer_product(Vec& x, Vec& y){
            Matrix result(y.size(), x.size());
                
            for (size_t i = 0; i < y.size(); i++){
                for (size_t j = 0; j < x.size(); j++)
                    result.row(i).data()[j] = y.data()[i] * x.data()[j];
            }
            return result;
        }
        Matrix outer_product(const Vec& x,  Vec& y){
            Matrix result(y.size(), x.size());
        
            for (size_t i = 0; i < y.size(); i++){
                for (size_t j = 0; j < x.size(); j++)
                    result.row(i).data()[j] = y.data()[i] * x.data()[j];
            }
            return result;
    }

};

//============================================================
// LayerTanh
//============================================================
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
    
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        
    }

};

//============================================================
// NeuralNet
//============================================================
class NeuralNet : public SupervisedLearner
{
    public:
        Vec weights;
        //Vec gradients;
        Vec act;
    
        vector<int> net_info;
        vector<Layer*> layers;
    
        // Constructors
        NeuralNet(vector<int> net_sets):
        weights(1){
            net_info = net_sets;
            // set layers
            int weights_len = 0;
            
            for (int i = 0; i < net_sets.size() - 1; i++){
                if(i % 2 == 0){
                    layers.push_back(new LayLinear(net_sets[i], net_sets[i + 1]));
                    weights_len += net_sets[i + 1] + net_sets[i + 1] * net_sets[i];
                    
                }else{
                    layers.push_back(new LayerTanh(net_sets[i + 1]));
                }
            }
            
            weights.resize(weights_len);

        }
    
        NeuralNet(size_t w) :
            weights(w){
        }
    
        NeuralNet(){}

    virtual ~NeuralNet(){
        for(size_t i = 0; i < layers.size(); i++)
            delete(layers[i]);
    }
    
    //----------------------------------------------------------
    virtual const char* name() { return "NeuralNet"; }
    
    //----------------------------------------------------------
    //// Hw1: Train
    virtual void train(const Matrix& features, const Matrix& labels){
        LayLinear tLayer(1, 1);
        tLayer.ordinary_least_squares(features, labels, weights);
    }
    
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
    // Predict
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
            
        act.copy(layers[layers.size() - 1]->activation);

        return act;
    }
    
    //----------------------------------------------------------
    // Backprop
    virtual void backProp(const Vec targets){
        // output layer blame
        layers[layers.size() - 1]->blame.copy(targets - layers[layers.size() - 1]->activation);
        ////cout << "output layer blame: "; layers[layers.size() - 1]->blame.print(); cout << endl;
        
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
    // Refine Weights (gradient decent)
    virtual void refineWeights(const Vec& net_input, const Vec& targets, double lr){
        Vec gradients(weights.size());
        
        // Stochastic
        for(int i = 0; i < 1; i++){
            gradients.fill(0.0);
        
            predict(net_input);
            backProp(targets);
            updateGradient(net_input, gradients);

            weights += gradients * lr;
        }
            
////////////////////////////////////
//        // Batch Maybe
//        Vec gradients(weights.size());
//
//        // refine per (mini) batch
//        gradients.fill(0,0);
//        int itr = 3;
//        for(int i = 0; i < itr; i++){
//            predict(net_input);
//            backProp(targets);
//            updateGradient(net_input, gradients);
//        }
//        weights += gradients * lr;
//        cout << "updated weights: "; weights.print(); cout << endl;
        
    }

};

//=============================================================
// Unit tests
//=============================================================

// test ordinary least squares
void test_ordinary_least_squares()
{
    Rand ra(2);
    int x_col = 2;
    int x_row = 3;
    int y_col = 1;
    int y_row = x_row;
    int weights_num = y_col + x_col * y_col;
    
    //Generate some random weights.
    Vec weights(weights_num);
    weights.fillUniform(ra, -1,1);
    
    
    //Generate a random feature matrix, X.
    Matrix X(x_row,x_col);
    for(int i =0; i < x_row;i++)
        X.row(i).fillUniform(ra, -1,1);
    
    //Use LinearLayer.activate to compute a corresponding label matrix, Y.
    Matrix Y(0, y_col);
    LayLinear *y = new LayLinear(weights.size(),y_col);
    for(int i=0;i<x_row;i++){
        y->activate(weights,X.row(i));
        Y.newRow().copy(y->activation);
    }
     //Add a little random noise to Y.
    Vec noise(y->activation.size());
    for(int i =0; i < y_row;i++){
        noise.fillUniform(ra,0.01,0.000001);
        //noise.print(std::cout);
        Y.row(i)+=noise;
    }    

    //Withhold the weights.
    Vec withHoldY(weights);

    //generate new weights
    y->ordinary_least_squares(X,Y,weights);
    
    //comparison between original weights and noised weights
    double threshold = 0.05;
    //std::cout << "sse:"<<sqrt(weights.squaredDistance(withHoldY))<<std::endl;
    if(sqrt(weights.squaredDistance(withHoldY)) < threshold)
        cout<<"Computed weights are close to the original weights"<<std::endl;
    else
        throw Ex("Computed weights differ too much from original weights");
}


//RMSE test
void RMSE(){
    // Load the training data
    string fn = "../bin/data/";
    Matrix trainFeatures;
    trainFeatures.loadARFF(fn+"housing_features.arff");
    Matrix trainLabels;
    trainLabels.loadARFF(fn+"house_lables.arff");
    
    // Train and measure the model
    NeuralNet learner;
    int iters = 50;
    for(int i = 0; i < iters; i++){
        double rmse = learner.NFoldCrossVal(trainFeatures, trainLabels,10,i);
        cout <<setprecision(14)<<"RMSE"<< " at " << "iters : " << i << " = " << rmse << "\n";
    }

}

// test layer update gradient
void testgradient(){
    LayLinear testlayer(1, 1);
    Vec gradient({0, 0, 1, 1, 2, 1, 1, 1});
    Vec x({3, 3, 3});
    testlayer.update_gradient(x, gradient);
    //gradient.print();
}


// test shuffle select
void test_shuffle(){
    
    vector<int> select_index(60000);
    int temp = 0;
    for(int i = 0; i < select_index.size(); i++){
        select_index[i] = temp;
        temp++;
    }
    //cout << "select vector: " << select_index << endl;
    int t = 0;
    random_shuffle(select_index.begin(), select_index.end());
    for(vector<int>::iterator it=select_index.begin(); it!=select_index.end(); ++it)
        t++;
    cout << t << endl;
}

//=======================================================================
// test refineWeights by comparing with OLS
//=======================================================================
void Test_refineWeights(){
    int x_col = 4;
    int x_row = 3;
    int y_col = 2;
    int y_row = x_row;
    
    Rand ra(2);
    Matrix X(x_row,x_col);
    for(int i =0; i < x_row;i++)
        X.row(i).fillUniform(ra, -1,1);
    
    Matrix Y(x_row, y_col);
    for(int i =0; i < x_row;i++)
        Y.row(i).fillUniform(ra, -1,1);
    
    cout<<"X"<<endl;
    X.print(cout);
    cout<<"Y"<<endl;
    Y.print(cout);
    
    vector<int> net_sets = {x_col,  y_col};
    NeuralNet bp_learner(net_sets);
    bp_learner.initWeights();
    
    cout << endl; cout << "initweights in refineWeights: ";
    bp_learner.weights.print(); cout << endl;
    
    for(int it = 0; it < 50000; it++){
        for(int i=0;i<x_row;i++)
            bp_learner.refineWeights(X[i], Y[i], 0.001);
  
    }
    
    cout << endl; cout << "weights in refineWeights: ";
    bp_learner.weights.print(); cout << endl;
    
    Vec Oweights(y_col + x_col * y_col);
    LayLinear *y = new LayLinear(Oweights.size(),10);
    y->ordinary_least_squares(X,Y,Oweights);
    
    cout << endl; cout << "weights in OLS: ";
    Oweights.print(); cout << endl;
    
}

//=======================================================================
// "debug spew" test
//=======================================================================
void h2_mini_test(){
    vector<int> net_sets = {1, 2, 2, 1};
    Vec feat_vec({0.3});
    Vec tar_vec({0.7});
    
    NeuralNet test_net(net_sets);
    test_net.weights.copy({0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3});
    
    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    
}

//=======================================================================
// MNIST TEST
//=======================================================================
int MNIST_test(NeuralNet& trained_model){
    // Load the testing data
    string fn = "../bin/data/";
    Matrix testFeatures;
    testFeatures.loadARFF(fn + "test_feat.arff");
    Matrix testLabels;
    testLabels.loadARFF(fn + "test_lab.arff");
    
    if(testFeatures.rows() != testLabels.rows())
        throw Ex("Mismatching number of rows");
    
    int predict;
    int target;
    int mis = 0;
    
    for(int i = 0; i < testFeatures.rows(); i++){
        const Vec pred = trained_model.predict(testFeatures[i]* (1.0 / 256));

        predict = pred.indexOfMax();
        target = testLabels[i][0];
        
        if(predict != target)
            mis++;
    }
    
    cout << "misclassification: " << mis << endl;
    
    return mis;
    
}

//=======================================================================
// MNIST train
//=======================================================================
void MNIST_train(){
    
    // Load the training data
    string fn = "../bin/data/";
    
    Matrix trainFeatures;
    trainFeatures.loadARFF(fn + "train_feat.arff");
    Matrix trainLabels;
    trainLabels.loadARFF(fn + "train_lab.arff");

    // Initailize the NeuralNet
    vector<int> net_sets = {784, 80, 80, 30, 30, 10, 10};
    
    NeuralNet bp_learner(net_sets);
    bp_learner.initWeights();
    

    // TRAIN begin
    cout << "before training: ";
    int mis = MNIST_test(bp_learner);
    
    while(mis > 350){
        if(trainFeatures.rows() != trainLabels.rows())
            throw Ex("Mismatching number of rows");
        else{
            // create index array
            vector<int> select_index(trainFeatures.rows());
            int temp = 0;
            for(int i = 0; i < select_index.size(); i++){
                select_index[i] = temp;
                temp++;
            }

            // loop over every pattern randomly
            random_shuffle(select_index.begin(), select_index.end());
            for(vector<int>::iterator it = select_index.begin(); it != select_index.end(); ++it){

                // create lab Vec for the NeuralNet output
                Vec lab(net_sets[net_sets.size() - 1]);
                for(int k = 0; k < lab.size(); k++){
                    if(k == trainLabels[*it][0])
                        lab[k] = 1.0;
                    else
                        lab[k] = 0.0;
                }

                bp_learner.refineWeights(trainFeatures[*it] * (1.0 / 256), lab, 0.03);
                
            }
            
  
        }//end else

        // Get Misclassification
        mis = MNIST_test(bp_learner);
        
    }//end learning loop
    
}


//=======================================================================
int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
    
	int ret = 1;
	try
	{
        //RMSE();
        //testlinear();
        //test_ordinary_least_squares();
        //testgradient();
        //test_shuffle();

        //Test_refineWeights();
        //h2_mini_test();
        
        MNIST_train();
        
        
		ret =0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}	
	

	 return ret;
}
