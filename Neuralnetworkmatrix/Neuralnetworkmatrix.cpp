// Neuralnetworkmatrix.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

// test_0402.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine
using namespace std;
//#include"Print.h"
/*
void update_weights(vector<vector<float>>& forward_weight,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    const vector<vector<float>>& input_slice,
    float learning_rate)
{
    int input_number = input_slice.size();
    int nodeCount = forward_weight[0].size();
    int sample_number = input_slice[0].size();

    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                forward_weight[j][k] -= learning_rate * error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]) * input_slice[j][n];
            }
        }
    }
}
*/
void update_weights_bias(vector<vector<float>>& forward_weight,
    vector<float>& biases,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    const vector<vector<float>>& input_slice,
    float learning_rate)
{
    int nodeCount = forward_weight[0].size();
    int sample_number = input_slice[0].size();

    for (int n = 0; n < sample_number; n++) {
        // 更新权重
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_slice.size(); j++) {
                forward_weight[j][k] -= learning_rate * error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]) * input_slice[j][n];
            }
        }

    }

    for (int k = 0; k < nodeCount; k++) {
        // 计算每个节点的误差总和
        float node_error_sum = 0.0;
        for (int j = 0; j < sample_number; j++) {
            node_error_sum += error[k][j];
        }
        biases[k] -= learning_rate * node_error_sum;
    }



}






void initialize_m_v(vector<vector<float>>& m,
    vector<vector<float>>& v,
    int num_rows,
    int num_cols) {


    m.resize(num_rows, vector<float>(num_cols, 0.0));
    v.resize(num_rows, vector<float>(num_cols, 0.0));
}

void adam_optimizer(vector<vector<float>>& weights,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    const vector<vector<float>>& input_slice,
    vector<vector<float>>& m,
    vector<vector<float>>& v,
    int t,
    float learning_rate,
    float beta1 = 0.9,
    float beta2 = 1,
    float epsilon = 1e-7)
{
    int input_number = input_slice.size();
    int nodeCount = weights[0].size();
    int sample_number = input_slice[0].size();

    // Update m and v
    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                // Compute gradient
                float gradient = error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]) * input_slice[j][n];
                //cout << "梯度：" << gradient << endl;
                // Update m and v
                m[j][k] = beta1 * m[j][k] + (1 - beta1) * gradient;
                v[j][k] = beta2 * v[j][k] + (1 - beta2) * pow(gradient, 2);
            }
        }
    }

    // Correct bias
    float beta1_t = pow(beta1, t);
    float beta2_t = pow(beta2, t);
    vector<vector<float>> m_hat(input_number, vector<float>(nodeCount));
    vector<vector<float>> v_hat(input_number, vector<float>(nodeCount));
    for (int j = 0; j < input_number; j++) {
        for (int k = 0; k < nodeCount; k++) {
            m_hat[j][k] = m[j][k] / (1 - beta1_t);
            v_hat[j][k] = v[j][k] / (1 - beta2_t);
        }
    }

    // Update weights
    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                weights[j][k] -= learning_rate * m_hat[j][k] / (sqrt(v_hat[j][k]) + epsilon);
            }
        }
    }


}

float leaky_relu(float x) {
    if (x > 0) {
        return x;
    }
    else {
        return 0.01 * x;
    }
}


void clipGradients(vector<vector<float>>& gradients, float threshold) {
    for (auto& row : gradients) {
        for (auto& val : row) {
            if (abs(val) > threshold) {
                val = (val > 0) ? threshold : -threshold;
            }
        }
    }
}


void adam_optimizer_bias(vector<vector<float>>& weights,
    vector<float> biases,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    const vector<vector<float>>& input_slice,
    vector<vector<float>>& m,
    vector<vector<float>>& v,
    int t,
    float learning_rate,
    float beta1 = 0.9,
    float beta2 = 0.99,
    float epsilon = 1e-7)
{
    int input_number = input_slice.size();
    int nodeCount = weights[0].size();
    int sample_number = input_slice[0].size();
    // Deep copy of m and v
    vector<vector<float>> m_buffer(input_number, vector<float>(nodeCount));
    vector<vector<float>> v_buffer(input_number, vector<float>(nodeCount));
    for (int i = 0; i < input_number; ++i) {
        copy(m[i].begin(), m[i].end(), m_buffer[i].begin());
        copy(v[i].begin(), v[i].end(), v_buffer[i].begin());
    }
    
    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                // Compute gradient
                float gradient = error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]) * input_slice[j][n];
                // Update m and v
                m_buffer[j][k] = beta1 * m_buffer[j][k] + (1 - beta1) * gradient;
                v_buffer[j][k] = beta2 * v_buffer[j][k] + (1 - beta2) * pow(gradient, 2);
                // Update weights
                weights[j][k] -= learning_rate * m_buffer[j][k] / (sqrt(v_buffer[j][k]) + epsilon);
            }
        }
    }

   
    // Copy m_buffer back to m
    for (int i = 0; i < input_number; ++i) {
        copy(m_buffer[i].begin(), m_buffer[i].end(), m[i].begin());
        copy(v_buffer[i].begin(), v_buffer[i].end(), v[i].begin());
    }

    

    for (int k = 0; k < nodeCount; k++) {
        // 计算每个节点的误差总和
        float node_error_sum = 0.0;
        for (int j = 0; j < sample_number; j++) {
            node_error_sum += error[k][j];
        }
        biases[k] -= learning_rate * node_error_sum;
    }
}





/*
void adam_optimizer_bias(vector<vector<float>>& weights,
    vector<float> biases,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    const vector<vector<float>>& input_slice,
    vector<vector<float>>& m,
    vector<vector<float>>& v,
    int t,
    float learning_rate,
    float clip_threshold,  // 添加裁剪阈值参数
    float beta1 = 0.9,
    float beta2 = 1,
    float epsilon = 1e-7)
{
    int input_number = input_slice.size();
    int nodeCount = weights[0].size();
    int sample_number = input_slice[0].size();

    // Update m and v
    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                // Compute gradient
                float gradient = error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]) * input_slice[j][n];
                // Update m and v
                m[j][k] = beta1 * m[j][k] + (1 - beta1) * abs(gradient);
                v[j][k] = beta2 * v[j][k] + (1 - beta2) * pow(gradient, 2);
            }
        }
    }

    // Clip gradients
    clipGradients(m, clip_threshold);
    clipGradients(v, clip_threshold);

    // Correct bias
    float beta1_t = pow(beta1, t);
    float beta2_t = pow(beta2, t);
    vector<vector<float>> m_hat(input_number, vector<float>(nodeCount));
    vector<vector<float>> v_hat(input_number, vector<float>(nodeCount));
    for (int j = 0; j < input_number; j++) {
        for (int k = 0; k < nodeCount; k++) {
            m_hat[j][k] = m[j][k] / (1 - beta1_t);
            v_hat[j][k] = v[j][k] / (1 - beta2_t);
        }
    }

    // Update weights
    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            for (int j = 0; j < input_number; j++) {
                weights[j][k] -= learning_rate * m_hat[j][k] / (sqrt(v_hat[j][k]) + epsilon);
            }
        }
    }

    for (int k = 0; k < nodeCount; k++) {
        // 计算每个节点的误差总和
        float node_error_sum = 0.0;
        for (int j = 0; j < sample_number; j++) {
            node_error_sum += error[k][j];
        }
        biases[k] -= learning_rate * node_error_sum;
    }
}
*/


vector<vector<float>> transposeMatrix(const vector<vector<float>>& matrix) {
    int rows = matrix.size();

    // 检查输入矩阵是否为空
    if (rows == 0) {
        cout << "Error: Input matrix is empty." << endl;
        return {};
    }

    int cols = matrix[0].size();

    // 检查每一行的列数是否相同
    for (int i = 1; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            cout << "Error: Inconsistent number of columns in the matrix." << endl;
            return {};
        }
    }

    // 创建一个新的矩阵来存储转置后的结果
    vector<vector<float>> result(cols, vector<float>(rows));

    // 遍历原始矩阵，并将元素按行列互换放置到新矩阵中
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}




/*
void update_bias(vector<float>& forward_bias,
    const vector<vector<float>>& layer_output,
    const vector<vector<float>>& error,
    float learning_rate)
{
    int nodeCount = forward_bias.size();
    int sample_number = layer_output[0].size();

    for (int n = 0; n < sample_number; n++) {
        for (int k = 0; k < nodeCount; k++) {
            forward_bias[k] -= learning_rate * error[k][n] * layer_output[k][n] * (1 - layer_output[k][n]);
        }
    }
}
*/




//计算均方误差
float calculateMSE(const vector<vector<float>>& errorArray) {
    float sumSquaredError = 0.0;
    int totalElements = 0;

    for (const auto& row : errorArray) {
        for (const auto& element : row) {
            float squaredError = element * element;
            sumSquaredError += squaredError;
            totalElements++;
        }
    }

    float mse = sumSquaredError / totalElements;
    return mse;
}

vector<vector<float>> calculateerror(const vector<vector<float>>& array1, const vector<vector<float>>& array2) {
    //size_t rows = array1.size();
        //size_t columns = array1[0].size();
        //vector<vector<float>> error(rows,vector<float>(columns));
    if (array1.size() == array2.size() && array1[0].size() == array2[0].size())
    {
        size_t rows = array1.size();
        size_t columns = array1[0].size();
        vector<vector<float>> error(rows, vector<float>(columns));
        for (size_t row = 0; row < rows; row++) {
            for (size_t column = 0; column < columns; column++) {
                error[row][column] = array1[row][column] - array2[row][column];
            }
        }
        return error;
    }
    else {
        cout << "数组维度不匹配" << endl;

    }



}
void print1DArray(const vector<float>& arr) {
    cout << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        cout << arr[i];
        if (i < arr.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

void print2DArray(const vector<vector<float>>& array) {

    for (const auto& row : array) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<vector<float>> initial_weight_Matrix(int inputCount, int nodeCount)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.1, 0.9); // 范围为 (0.1, 0.9) 的随机浮点数分布
    vector<vector<float>> weight_matrix(inputCount, vector<float>(nodeCount));

    for (int i = 0; i < inputCount; ++i) {
        for (int j = 0; j < nodeCount; ++j) {
            weight_matrix[i][j] = static_cast<float>(dis(gen));
        }
    }

    return weight_matrix;
}

vector<float> initial_bias_Matrix(int nodeCount)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.1, 0.9); // 范围为 (0.1, 0.9) 的随机浮点数分布
    vector<float> bias_matrix(nodeCount);

    for (int i = 0; i < nodeCount; i++) {
        bias_matrix[i] = static_cast<float>(dis(gen));
    }

    return bias_matrix;
}

//激活函数
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float my_tanh(float x) {
    //return tanh(x);

    return tanh(x/3);//3
}

float xigmoid(float x) {
    if (x > 0) {
        return sigmoid(1.3 * (exp(1.3 * x) - 1));
        
    }
    else
        return sigmoid(1.3 * (1 - exp(-1.3 * x)));
}


/*初始化前馈输出层，返回权值矩阵，偏置矩阵，以及节点输出结果*/
tuple<vector<vector<float>>, vector<float>, vector<vector<float>>> initial_neurallayer_output(const vector<vector<float>>& input_slice, int nodeCount, int function_type)//选择激活函数1表示sigmoid。2表示tanh
{
    int input_number = input_slice.size();//输入变量维数，假如是个二维矩阵
    int sample_number = input_slice[0].size();//样本数量
    //cout << input_number << " " << sample_number << endl;
    vector<vector<float>> weightmatrix = initial_weight_Matrix(input_number, nodeCount);//权重矩阵初始化
    vector<float> biasmatrix = initial_bias_Matrix(nodeCount);
    vector<vector<float>> layer_output(nodeCount, vector<float>(sample_number, 0));

    for (int n = 0; n < sample_number; n++)//每个样本切片对应的神经网络输出
    {
        for (int i = 0; i < nodeCount; i++)//神经网络节点输出
        {
            layer_output[i][n] = biasmatrix[i];
            for (int j = 0; j < input_number; j++) //行表示输入，加权过程
            {
                layer_output[i][n] += weightmatrix[j][i] * input_slice[j][n];
            }
            switch (function_type)//激活函数
            {
            case 1:
                layer_output[i][n] = sigmoid(layer_output[i][n]);
                break;
            case 2:
                layer_output[i][n] = my_tanh(layer_output[i][n]);
                break;
            case 3:
                layer_output[i][n] = xigmoid(layer_output[i][n]);
                break;

            }
        }

    }

    return make_tuple(layer_output, biasmatrix, weightmatrix);
}

vector<vector<float>> training_forward_neurallayer_output(vector<vector<float>>& input_slice,
    vector<vector<float>>& forward_weight,
    vector<float>& forward_bias,
    int nodeCount,
    int function_type)
{
    int input_number = input_slice.size(); // 上一层节点数
    int sample_number = input_slice[0].size(); // 样本数量
    vector<vector<float>> layer_output(nodeCount, vector<float>(sample_number, 0));

    // 将权重矩阵转置，方便矩阵乘法
    vector<vector<float>> weightmatrix_T=transposeMatrix(forward_weight);
    

    // 矩阵乘法
    for (int n = 0; n < sample_number; n++) {
        for (int i = 0; i < nodeCount; i++) {
            layer_output[i][n] = forward_bias[i];
            for (int j = 0; j < input_number; j++) {
                layer_output[i][n] += weightmatrix_T[i][j] * input_slice[j][n];
            }
        }
    }

    // 应用激活函数
    for (int n = 0; n < sample_number; n++) {
        for (int i = 0; i < nodeCount; i++) {
            switch (function_type) {
            case 1:
                layer_output[i][n] = sigmoid(layer_output[i][n]);
                break;
            case 2:
                layer_output[i][n] = my_tanh(layer_output[i][n]);
                break;
            case 3:
                layer_output[i][n] = xigmoid(layer_output[i][n]);
                break;
            case 4:
                layer_output[i][n] = leaky_relu(layer_output[i][n]);
                break;
            }
        }
    }

    return layer_output;
}
/*
vector<vector<float>> training_forward_neurallayer_output(vector<vector<float>>& input_slice,
    vector<vector<float>>& forward_weight,
    vector<float>& forward_bias,
    int nodeCount,
    int function_type)
{
    int input_number = input_slice.size();//上一层节点数
    int sample_number = input_slice[0].size();//样本数量
    vector<vector<float>> weightmatrix = forward_weight;//权重矩阵初始化
    vector<float> biasmatrix = forward_bias;
    vector<vector<float>> layer_output(nodeCount, vector<float>(sample_number, 0));

    for (int n = 0; n < sample_number; n++)//每个样本切片对应的神经网络输出
    {
        for (int i = 0; i < nodeCount; i++)//神经网络节点输出
        {
            layer_output[i][n] = biasmatrix[i];
            for (int j = 0; j < input_number; j++) //行表示输入，加权过程
            {
                layer_output[i][n] += weightmatrix[j][i] * input_slice[j][n];
            }
            switch (function_type)//激活函数
            {
            case 1:
                layer_output[i][n] = sigmoid(layer_output[i][n]);
                break;
            case 2:
                layer_output[i][n] = my_tanh(layer_output[i][n]);
                break;
            case 3:
                layer_output[i][n] = xigmoid(layer_output[i][n]);
                break;
            }
        }

    }

    return  layer_output;
}

*/




/*
int main()
{

    const vector<vector<float>> initial_data= {
    {(float)+6.92191844e-01, (float)+9.95272036e-01, (float)+9.86803804e-01, (float)+8.32296560e-01, (float)+6.31590248e-01, },
    {(float)+6.91496756e-01, (float)+9.95272036e-01, (float)+9.86027557e-01, (float)+8.32042656e-01, (float)+6.30731369e-01, },
    {(float)+6.90801668e-01, (float)+9.95272036e-01, (float)+9.85251310e-01, (float)+8.31661800e-01, (float)+6.29872489e-01, },
    {(float)+6.90106580e-01, (float)+9.95295795e-01, (float)+9.84475063e-01, (float)+8.31407896e-01, (float)+6.28997093e-01, },
    {(float)+6.89411492e-01, (float)+9.95295795e-01, (float)+9.83698816e-01, (float)+8.31027041e-01, (float)+6.28138214e-01, },
    {(float)+6.88716404e-01, (float)+9.95295795e-01, (float)+9.82728508e-01, (float)+8.30773137e-01, (float)+6.27279334e-01, },
    {(float)+6.88021316e-01, (float)+9.95319553e-01, (float)+9.81952261e-01, (float)+8.30519233e-01, (float)+6.26420455e-01, },
    {(float)+6.87326228e-01, (float)+9.95319553e-01, (float)+9.81176014e-01, (float)+8.30138378e-01, (float)+6.25561575e-01, },
    {(float)+6.86631140e-01, (float)+9.95319553e-01, (float)+9.80399767e-01, (float)+8.29884474e-01, (float)+6.24702696e-01, },
    {(float)+6.85936052e-01, (float)+9.95343312e-01, (float)+9.79623520e-01, (float)+8.29630570e-01, (float)+6.23843816e-01, },
    {(float)+6.85240964e-01, (float)+9.95343312e-01, (float)+9.78653212e-01, (float)+8.29249714e-01, (float)+6.22968420e-01, },
    {(float)+6.84545876e-01, (float)+9.95343312e-01, (float)+9.77876965e-01, (float)+8.28995811e-01, (float)+6.22109540e-01, },
    {(float)+6.83850788e-01, (float)+9.95367071e-01, (float)+9.77100718e-01, (float)+8.28614955e-01, (float)+6.21250661e-01, },
    {(float)+6.83155700e-01, (float)+9.95367071e-01, (float)+9.76324471e-01, (float)+8.28361051e-01, (float)+6.20391781e-01, },
    {(float)+6.82460612e-01, (float)+9.95367071e-01, (float)+9.75548224e-01, (float)+8.28107147e-01, (float)+6.19532902e-01, },
    {(float)+6.81765524e-01, (float)+9.95390829e-01, (float)+9.74577916e-01, (float)+8.27726292e-01, (float)+6.18674022e-01, },
    {(float)+6.81070436e-01, (float)+9.95390829e-01, (float)+9.73801669e-01, (float)+8.27472388e-01, (float)+6.17815143e-01, },
    {(float)+6.80375348e-01, (float)+9.95390829e-01, (float)+9.73025422e-01, (float)+8.27091532e-01, (float)+6.16939746e-01, },
    {(float)+6.79680259e-01, (float)+9.95414588e-01, (float)+9.72249175e-01, (float)+8.26837629e-01, (float)+6.16080867e-01, },
    {(float)+6.78985171e-01, (float)+9.95414588e-01, (float)+9.71472928e-01, (float)+8.26583725e-01, (float)+6.15221987e-01, },
    {(float)+6.78290083e-01, (float)+9.95414588e-01, (float)+9.70502620e-01, (float)+8.26202869e-01, (float)+6.14363108e-01, },
    {(float)+6.77594995e-01, (float)+9.95438346e-01, (float)+9.69726373e-01, (float)+8.25948965e-01, (float)+6.13504228e-01, },
    {(float)+6.76899907e-01, (float)+9.95438346e-01, (float)+9.68950126e-01, (float)+8.25695062e-01, (float)+6.12645349e-01, },
    {(float)+6.76204819e-01, (float)+9.95438346e-01, (float)+9.68173879e-01, (float)+8.25314206e-01, (float)+6.11786469e-01, },
    {(float)+6.75509731e-01, (float)+9.95462105e-01, (float)+9.67397632e-01, (float)+8.25060302e-01, (float)+6.10927590e-01, },
    {(float)+6.74814643e-01, (float)+9.95462105e-01, (float)+9.66427324e-01, (float)+8.24679446e-01, (float)+6.10052193e-01, },
    {(float)+6.74119555e-01, (float)+9.95462105e-01, (float)+9.65651077e-01, (float)+8.24425543e-01, (float)+6.09193314e-01, },
    {(float)+6.73424467e-01, (float)+9.95485864e-01, (float)+9.64874830e-01, (float)+8.24171639e-01, (float)+6.08334434e-01, },
    {(float)+6.72729379e-01, (float)+9.95485864e-01, (float)+9.64098583e-01, (float)+8.23790783e-01, (float)+6.07475555e-01, },
    {(float)+6.72034291e-01, (float)+9.95485864e-01, (float)+9.63322337e-01, (float)+8.23536880e-01, (float)+6.06616675e-01, },
    {(float)+6.71223355e-01, (float)+9.95509622e-01, (float)+9.62546090e-01, (float)+8.23156024e-01, (float)+6.05757796e-01, },
    {(float)+6.70528267e-01, (float)+9.95509622e-01, (float)+9.61575781e-01, (float)+8.22902120e-01, (float)+6.04898916e-01, },
    {(float)+6.69833179e-01, (float)+9.95509622e-01, (float)+9.60799534e-01, (float)+8.22648216e-01, (float)+6.04023520e-01, },
    {(float)+6.69138091e-01, (float)+9.95533381e-01, (float)+9.60023287e-01, (float)+8.22267361e-01, (float)+6.03164641e-01, },
    {(float)+6.68443003e-01, (float)+9.95533381e-01, (float)+9.59247041e-01, (float)+8.22013457e-01, (float)+6.02305761e-01, },
    {(float)+6.67747915e-01, (float)+9.95557139e-01, (float)+9.58470794e-01, (float)+8.21759553e-01, (float)+6.01446882e-01, },
    {(float)+6.67052827e-01, (float)+9.95557139e-01, (float)+9.57500485e-01, (float)+8.21378697e-01, (float)+6.00588002e-01, },
    {(float)+6.66357739e-01, (float)+9.95557139e-01, (float)+9.56724238e-01, (float)+8.21124794e-01, (float)+5.99729123e-01, },
    {(float)+6.65662651e-01, (float)+9.95580898e-01, (float)+9.55947991e-01, (float)+8.20743938e-01, (float)+5.98870243e-01, },
    {(float)+6.64967563e-01, (float)+9.95580898e-01, (float)+9.55171745e-01, (float)+8.20490034e-01, (float)+5.97994847e-01, },
    {(float)+6.64272475e-01, (float)+9.95580898e-01, (float)+9.54395498e-01, (float)+8.20236131e-01, (float)+5.97135967e-01, },
    {(float)+6.63577386e-01, (float)+9.95604657e-01, (float)+9.53425189e-01, (float)+8.19855275e-01, (float)+5.96277088e-01, },
    {(float)+6.62882298e-01, (float)+9.95604657e-01, (float)+9.52648942e-01, (float)+8.19601371e-01, (float)+5.95418208e-01, },
    {(float)+6.62187210e-01, (float)+9.95604657e-01, (float)+9.51872696e-01, (float)+8.19347467e-01, (float)+5.94559329e-01, },
    {(float)+6.61492122e-01, (float)+9.95628415e-01, (float)+9.51096449e-01, (float)+8.18966612e-01, (float)+5.93700449e-01, },
    {(float)+6.60797034e-01, (float)+9.95628415e-01, (float)+9.50320202e-01, (float)+8.18712708e-01, (float)+5.92841570e-01, },
    {(float)+6.60101946e-01, (float)+9.95628415e-01, (float)+9.49349893e-01, (float)+8.18331852e-01, (float)+5.91966173e-01, },
    {(float)+6.59406858e-01, (float)+9.95652174e-01, (float)+9.48573646e-01, (float)+8.18077948e-01, (float)+5.91107294e-01, },
    {(float)+6.58711770e-01, (float)+9.95652174e-01, (float)+9.47797400e-01, (float)+8.17824045e-01, (float)+5.90248414e-01, },
    {(float)+6.58016682e-01, (float)+9.95652174e-01, (float)+9.47021153e-01, (float)+8.17443189e-01, (float)+5.89389535e-01, },
    {(float)+6.57321594e-01, (float)+9.95675933e-01, (float)+9.46244906e-01, (float)+8.17189285e-01, (float)+5.88530655e-01, },
    {(float)+6.56626506e-01, (float)+9.95675933e-01, (float)+9.45274597e-01, (float)+8.16808430e-01, (float)+5.87671776e-01, },
    {(float)+6.55931418e-01, (float)+9.95675933e-01, (float)+9.44498350e-01, (float)+8.16554526e-01, (float)+5.86812896e-01, },
    {(float)+6.55236330e-01, (float)+9.95699691e-01, (float)+9.43722104e-01, (float)+8.16300622e-01, (float)+5.85937500e-01, },
    {(float)+6.54541242e-01, (float)+9.95699691e-01, (float)+9.42945857e-01, (float)+8.15919766e-01, (float)+5.85078621e-01, },
    {(float)+6.53846154e-01, (float)+9.95699691e-01, (float)+9.42169610e-01, (float)+8.15665863e-01, (float)+5.84219741e-01, },
    {(float)+6.53151066e-01, (float)+9.95723450e-01, (float)+9.41199301e-01, (float)+8.15411959e-01, (float)+5.83360862e-01, },
    {(float)+6.52455978e-01, (float)+9.95723450e-01, (float)+9.40423055e-01, (float)+8.15031103e-01, (float)+5.82501982e-01, },
    {(float)+6.51760890e-01, (float)+9.95723450e-01, (float)+9.39646808e-01, (float)+8.14777199e-01, (float)+5.81643103e-01, },
    {(float)+6.51065802e-01, (float)+9.95747208e-01, (float)+9.38870561e-01, (float)+8.14396344e-01, (float)+5.80784223e-01, },
    {(float)+6.50370714e-01, (float)+9.95747208e-01, (float)+9.38094314e-01, (float)+8.14142440e-01, (float)+5.79908827e-01, },
    {(float)+6.49675626e-01, (float)+9.95747208e-01, (float)+9.37124005e-01, (float)+8.13888536e-01, (float)+5.79049947e-01, },
    {(float)+6.48980538e-01, (float)+9.95770967e-01, (float)+9.36347759e-01, (float)+8.13507681e-01, (float)+5.78191068e-01, },
    {(float)+6.48285449e-01, (float)+9.95770967e-01, (float)+9.35571512e-01, (float)+8.13253777e-01, (float)+5.77332188e-01, },
    {(float)+6.47590361e-01, (float)+9.95770967e-01, (float)+9.34795265e-01, (float)+8.12872921e-01, (float)+5.76473309e-01, },
    {(float)+6.46895273e-01, (float)+9.95794726e-01, (float)+9.34019018e-01, (float)+8.12619017e-01, (float)+5.75614429e-01, },
    {(float)+6.46200185e-01, (float)+9.95794726e-01, (float)+9.33048709e-01, (float)+8.12365114e-01, (float)+5.74755550e-01, },
    {(float)+6.45505097e-01, (float)+9.95794726e-01, (float)+9.32272463e-01, (float)+8.11984258e-01, (float)+5.73880153e-01, },
    {(float)+6.44810009e-01, (float)+9.95818484e-01, (float)+9.31496216e-01, (float)+8.11730354e-01, (float)+5.73021274e-01, },
    {(float)+6.44114921e-01, (float)+9.95818484e-01, (float)+9.30719969e-01, (float)+8.11476450e-01, (float)+5.72162394e-01, },
    {(float)+6.43419833e-01, (float)+9.95818484e-01, (float)+9.29943722e-01, (float)+8.11095595e-01, (float)+5.71303515e-01, },
    {(float)+6.42724745e-01, (float)+9.95842243e-01, (float)+9.28973414e-01, (float)+8.10841691e-01, (float)+5.70444635e-01, },
    {(float)+6.42029657e-01, (float)+9.95842243e-01, (float)+9.28197167e-01, (float)+8.10460835e-01, (float)+5.69585756e-01, },
    {(float)+6.41334569e-01, (float)+9.95842243e-01, (float)+9.27420920e-01, (float)+8.10206932e-01, (float)+5.68726876e-01, },
    {(float)+6.40639481e-01, (float)+9.95866001e-01, (float)+9.26644673e-01, (float)+8.09953028e-01, (float)+5.67867997e-01, },
    {(float)+6.39944393e-01, (float)+9.95866001e-01, (float)+9.25868426e-01, (float)+8.09572172e-01, (float)+5.66992600e-01, },
    {(float)+6.39249305e-01, (float)+9.95866001e-01, (float)+9.24898118e-01, (float)+8.09318268e-01, (float)+5.66133721e-01, },
    {(float)+6.38554217e-01, (float)+9.95889760e-01, (float)+9.24121871e-01, (float)+8.09064365e-01, (float)+5.65274841e-01, },
    {(float)+6.37859129e-01, (float)+9.95889760e-01, (float)+9.23345624e-01, (float)+8.08683509e-01, (float)+5.64415962e-01, },
    {(float)+6.37164041e-01, (float)+9.95889760e-01, (float)+9.22569377e-01, (float)+8.08429605e-01, (float)+5.63557082e-01, },
    {(float)+6.36468953e-01, (float)+9.95913519e-01, (float)+9.21793130e-01, (float)+8.08048750e-01, (float)+5.62698203e-01, },
    {(float)+6.35773865e-01, (float)+9.95913519e-01, (float)+9.20822822e-01, (float)+8.07794846e-01, (float)+5.61839323e-01, },
    {(float)+6.35078777e-01, (float)+9.95913519e-01, (float)+9.20046575e-01, (float)+8.07540942e-01, (float)+5.60963927e-01, },
    {(float)+6.34383689e-01, (float)+9.95937277e-01, (float)+9.19270328e-01, (float)+8.07160086e-01, (float)+5.60105048e-01, },
    {(float)+6.33688601e-01, (float)+9.95937277e-01, (float)+9.18494081e-01, (float)+8.06906183e-01, (float)+5.59246168e-01, },
    {(float)+6.32993513e-01, (float)+9.95937277e-01, (float)+9.17717834e-01, (float)+8.06525327e-01, (float)+5.58387289e-01, },
    {(float)+6.32298424e-01, (float)+9.95961036e-01, (float)+9.16747526e-01, (float)+8.06271423e-01, (float)+5.57528409e-01, },
    {(float)+6.31603336e-01, (float)+9.95961036e-01, (float)+9.15971279e-01, (float)+8.06017519e-01, (float)+5.56669530e-01, },
    {(float)+6.30908248e-01, (float)+9.95961036e-01, (float)+9.15195032e-01, (float)+8.05636664e-01, (float)+5.55810650e-01, },
    {(float)+6.30213160e-01, (float)+9.95984794e-01, (float)+9.14418785e-01, (float)+8.05382760e-01, (float)+5.54935254e-01, },
    {(float)+6.29518072e-01, (float)+9.95984794e-01, (float)+9.13642538e-01, (float)+8.05128856e-01, (float)+5.54076374e-01, },
    {(float)+6.28822984e-01, (float)+9.95984794e-01, (float)+9.12672230e-01, (float)+8.04748001e-01, (float)+5.53217495e-01, },
    {(float)+6.28127896e-01, (float)+9.96008553e-01, (float)+9.11895983e-01, (float)+8.04494097e-01, (float)+5.52358615e-01, },
    {(float)+6.27432808e-01, (float)+9.96008553e-01, (float)+9.11119736e-01, (float)+8.04113241e-01, (float)+5.51499736e-01, },
    {(float)+6.26737720e-01, (float)+9.96008553e-01, (float)+9.10343489e-01, (float)+8.03859337e-01, (float)+5.50640856e-01, },
    {(float)+6.26042632e-01, (float)+9.96032312e-01, (float)+9.09567242e-01, (float)+8.03605434e-01, (float)+5.49781977e-01, },
    {(float)+6.25347544e-01, (float)+9.96032312e-01, (float)+9.08790996e-01, (float)+8.03224578e-01, (float)+5.48906580e-01, },
    {(float)+6.24652456e-01, (float)+9.96032312e-01, (float)+9.07820687e-01, (float)+8.02970674e-01, (float)+5.48047701e-01, },
    {(float)+6.23957368e-01, (float)+9.96056070e-01, (float)+9.07044440e-01, (float)+8.02589818e-01, (float)+5.47188821e-01, },
    {(float)+6.23262280e-01, (float)+9.96056070e-01, (float)+9.06268193e-01, (float)+8.02335915e-01, (float)+5.46329942e-01, },
    {(float)+6.22567192e-01, (float)+9.96056070e-01, (float)+9.05491946e-01, (float)+8.02082011e-01, (float)+5.45471062e-01, },
    {(float)+6.21872104e-01, (float)+9.96079829e-01, (float)+9.04715700e-01, (float)+8.01701155e-01, (float)+5.44612183e-01, },
    {(float)+6.21177016e-01, (float)+9.96079829e-01, (float)+9.03745391e-01, (float)+8.01447251e-01, (float)+5.43753303e-01, },
    {(float)+6.20481928e-01, (float)+9.96079829e-01, (float)+9.02969144e-01, (float)+8.01193348e-01, (float)+5.42877907e-01, },
    {(float)+6.19786840e-01, (float)+9.96103588e-01, (float)+9.02192897e-01, (float)+8.00812492e-01, (float)+5.42019027e-01, },
    {(float)+6.18975904e-01, (float)+9.96103588e-01, (float)+9.01416650e-01, (float)+8.00558588e-01, (float)+5.41160148e-01, },
    {(float)+6.18280816e-01, (float)+9.96103588e-01, (float)+9.00640404e-01, (float)+8.00177733e-01, (float)+5.40301268e-01, },
    {(float)+6.17585728e-01, (float)+9.96127346e-01, (float)+8.99670095e-01, (float)+7.99923829e-01, (float)+5.39442389e-01, },
    {(float)+6.16890639e-01, (float)+9.96127346e-01, (float)+8.98893848e-01, (float)+7.99669925e-01, (float)+5.38583510e-01, },
    {(float)+6.16195551e-01, (float)+9.96127346e-01, (float)+8.98117601e-01, (float)+7.99289069e-01, (float)+5.37724630e-01, },
    {(float)+6.15500463e-01, (float)+9.96151105e-01, (float)+8.97341355e-01, (float)+7.99035166e-01, (float)+5.36849234e-01, },
    {(float)+6.14805375e-01, (float)+9.96151105e-01, (float)+8.96565108e-01, (float)+7.98781262e-01, (float)+5.35990354e-01, },
    {(float)+6.14110287e-01, (float)+9.96151105e-01, (float)+8.95594799e-01, (float)+7.98400406e-01, (float)+5.35131475e-01, },
    {(float)+6.13415199e-01, (float)+9.96174863e-01, (float)+8.94818552e-01, (float)+7.98146502e-01, (float)+5.34272595e-01, },
    {(float)+6.12720111e-01, (float)+9.96174863e-01, (float)+8.94042305e-01, (float)+7.97765647e-01, (float)+5.33413716e-01, },
    {(float)+6.12025023e-01, (float)+9.96174863e-01, (float)+8.93266059e-01, (float)+7.97511743e-01, (float)+5.32554836e-01, },
    {(float)+6.11329935e-01, (float)+9.96198622e-01, (float)+8.92489812e-01, (float)+7.97257839e-01, (float)+5.31695957e-01, },
    {(float)+6.10634847e-01, (float)+9.96198622e-01, (float)+8.91519503e-01, (float)+7.96876984e-01, (float)+5.30820560e-01, },
    {(float)+6.09939759e-01, (float)+9.96198622e-01, (float)+8.90743256e-01, (float)+7.96623080e-01, (float)+5.29961681e-01, },
    {(float)+6.09244671e-01, (float)+9.96222381e-01, (float)+8.89967010e-01, (float)+7.96242224e-01, (float)+5.29102801e-01, },
    {(float)+6.08549583e-01, (float)+9.96222381e-01, (float)+8.89190763e-01, (float)+7.95988320e-01, (float)+5.28243922e-01, },
    {(float)+6.07854495e-01, (float)+9.96246139e-01, (float)+8.88414516e-01, (float)+7.95734417e-01, (float)+5.27385042e-01, },
    {(float)+6.07159407e-01, (float)+9.96246139e-01, (float)+8.87444207e-01, (float)+7.95353561e-01, (float)+5.26526163e-01, },
    {(float)+6.06464319e-01, (float)+9.96246139e-01, (float)+8.86667960e-01, (float)+7.95099657e-01, (float)+5.25667283e-01, },
    {(float)+6.05769231e-01, (float)+9.96269898e-01, (float)+8.85891714e-01, (float)+7.94845753e-01, (float)+5.24791887e-01, },
    {(float)+6.05074143e-01, (float)+9.96269898e-01, (float)+8.85115467e-01, (float)+7.94464898e-01, (float)+5.23933007e-01, },
    {(float)+6.04379055e-01, (float)+9.96269898e-01, (float)+8.84339220e-01, (float)+7.94210994e-01, (float)+5.23074128e-01, },
    {(float)+6.03683967e-01, (float)+9.96293656e-01, (float)+8.83368911e-01, (float)+7.93830138e-01, (float)+5.22215248e-01, },
    {(float)+6.02988879e-01, (float)+9.96293656e-01, (float)+8.82592664e-01, (float)+7.93576235e-01, (float)+5.21356369e-01, },
    {(float)+6.02293791e-01, (float)+9.96293656e-01, (float)+8.81816418e-01, (float)+7.93322331e-01, (float)+5.20497489e-01, },
    {(float)+6.01598703e-01, (float)+9.96317415e-01, (float)+8.81040171e-01, (float)+7.92941475e-01, (float)+5.19638610e-01, },
    {(float)+6.00903614e-01, (float)+9.96317415e-01, (float)+8.80263924e-01, (float)+7.92687571e-01, (float)+5.18779730e-01, },
    {(float)+6.00208526e-01, (float)+9.96317415e-01, (float)+8.79293615e-01, (float)+7.92306716e-01, (float)+5.17904334e-01, },
    {(float)+5.99513438e-01, (float)+9.96341174e-01, (float)+8.78517369e-01, (float)+7.92052812e-01, (float)+5.17045455e-01, },
    {(float)+5.98818350e-01, (float)+9.96341174e-01, (float)+8.77741122e-01, (float)+7.91798908e-01, (float)+5.16186575e-01, },
    {(float)+5.98123262e-01, (float)+9.96341174e-01, (float)+8.76964875e-01, (float)+7.91418053e-01, (float)+5.15327696e-01, },
    {(float)+5.97428174e-01, (float)+9.96364932e-01, (float)+8.76188628e-01, (float)+7.91164149e-01, (float)+5.14468816e-01, },
    {(float)+5.96733086e-01, (float)+9.96364932e-01, (float)+8.75218319e-01, (float)+7.90910245e-01, (float)+5.13609937e-01, },
    {(float)+5.96037998e-01, (float)+9.96364932e-01, (float)+8.74442073e-01, (float)+7.90529389e-01, (float)+5.12751057e-01, },
    {(float)+5.95342910e-01, (float)+9.96388691e-01, (float)+8.73665826e-01, (float)+7.90275486e-01, (float)+5.11875661e-01, },
    {(float)+5.94647822e-01, (float)+9.96388691e-01, (float)+8.72889579e-01, (float)+7.89894630e-01, (float)+5.11016781e-01, },
    {(float)+5.93952734e-01, (float)+9.96388691e-01, (float)+8.72113332e-01, (float)+7.89640726e-01, (float)+5.10157902e-01, },
    {(float)+5.93257646e-01, (float)+9.96412450e-01, (float)+8.71143023e-01, (float)+7.89386822e-01, (float)+5.09299022e-01, },
    {(float)+5.92562558e-01, (float)+9.96412450e-01, (float)+8.70366777e-01, (float)+7.89005967e-01, (float)+5.08440143e-01, },
    {(float)+5.91867470e-01, (float)+9.96412450e-01, (float)+8.69590530e-01, (float)+7.88752063e-01, (float)+5.07581263e-01, },
    {(float)+5.91172382e-01, (float)+9.96436208e-01, (float)+8.68814283e-01, (float)+7.88498159e-01, (float)+5.06722384e-01, },
    {(float)+5.90477294e-01, (float)+9.96436208e-01, (float)+8.68038036e-01, (float)+7.88117304e-01, (float)+5.05846987e-01, },
    {(float)+5.89782206e-01, (float)+9.96436208e-01, (float)+8.67067728e-01, (float)+7.87863400e-01, (float)+5.04988108e-01, },
    {(float)+5.89087118e-01, (float)+9.96459967e-01, (float)+8.66291481e-01, (float)+7.87482544e-01, (float)+5.04129228e-01, },
    {(float)+5.88392030e-01, (float)+9.96459967e-01, (float)+8.65515234e-01, (float)+7.87228640e-01, (float)+5.03270349e-01, },
    {(float)+5.87696942e-01, (float)+9.96459967e-01, (float)+8.64738987e-01, (float)+7.86974737e-01, (float)+5.02411469e-01, },
    {(float)+5.87001854e-01, (float)+9.96483725e-01, (float)+8.63962740e-01, (float)+7.86593881e-01, (float)+5.01552590e-01, },
    {(float)+5.86306766e-01, (float)+9.96483725e-01, (float)+8.62992432e-01, (float)+7.86339977e-01, (float)+5.00693710e-01, },
    {(float)+5.85611677e-01, (float)+9.96483725e-01, (float)+8.62216185e-01, (float)+7.85959121e-01, (float)+4.99818314e-01, },
    {(float)+5.84916589e-01, (float)+9.96507484e-01, (float)+8.61439938e-01, (float)+7.85705218e-01, (float)+4.98959434e-01, },
    {(float)+5.84221501e-01, (float)+9.96507484e-01, (float)+8.60663691e-01, (float)+7.85451314e-01, (float)+4.98100555e-01, },
    {(float)+5.83526413e-01, (float)+9.96507484e-01, (float)+8.59887444e-01, (float)+7.85070458e-01, (float)+4.97241675e-01, },
    {(float)+5.82831325e-01, (float)+9.96531243e-01, (float)+8.58917136e-01, (float)+7.84816555e-01, (float)+4.96382796e-01, },
    {(float)+5.82136237e-01, (float)+9.96531243e-01, (float)+8.58140889e-01, (float)+7.84562651e-01, (float)+4.95523916e-01, },
    {(float)+5.81441149e-01, (float)+9.96531243e-01, (float)+8.57364642e-01, (float)+7.84181795e-01, (float)+4.94665037e-01, },
    {(float)+5.80746061e-01, (float)+9.96555001e-01, (float)+8.56588395e-01, (float)+7.83927891e-01, (float)+4.93789641e-01, },
    {(float)+5.80050973e-01, (float)+9.96555001e-01, (float)+8.55812148e-01, (float)+7.83547036e-01, (float)+4.92930761e-01, },
    {(float)+5.79355885e-01, (float)+9.96555001e-01, (float)+8.55035901e-01, (float)+7.83293132e-01, (float)+4.92071882e-01, },
    {(float)+5.78660797e-01, (float)+9.96578760e-01, (float)+8.54065593e-01, (float)+7.83039228e-01, (float)+4.91213002e-01, },
    {(float)+5.77965709e-01, (float)+9.96578760e-01, (float)+8.53289346e-01, (float)+7.82658372e-01, (float)+4.90354123e-01, },
    {(float)+5.77270621e-01, (float)+9.96578760e-01, (float)+8.52513099e-01, (float)+7.82404469e-01, (float)+4.89495243e-01, },
    {(float)+5.76575533e-01, (float)+9.96602518e-01, (float)+8.51736852e-01, (float)+7.82023613e-01, (float)+4.88636364e-01, },
    {(float)+5.75880445e-01, (float)+9.96602518e-01, (float)+8.50960605e-01, (float)+7.81769709e-01, (float)+4.87760967e-01, },
    {(float)+5.75185357e-01, (float)+9.96602518e-01, (float)+8.49990297e-01, (float)+7.81515806e-01, (float)+4.86902088e-01, },
    {(float)+5.74490269e-01, (float)+9.96626277e-01, (float)+8.49214050e-01, (float)+7.81134950e-01, (float)+4.86043208e-01, },
    {(float)+5.73795181e-01, (float)+9.96626277e-01, (float)+8.48437803e-01, (float)+7.80881046e-01, (float)+4.85184329e-01, },
    {(float)+5.73100093e-01, (float)+9.96626277e-01, (float)+8.47661556e-01, (float)+7.80627142e-01, (float)+4.84325449e-01, },
    {(float)+5.72405005e-01, (float)+9.96650036e-01, (float)+8.46885310e-01, (float)+7.80246287e-01, (float)+4.83466570e-01, },
    {(float)+5.71709917e-01, (float)+9.96650036e-01, (float)+8.45915001e-01, (float)+7.79992383e-01, (float)+4.82607690e-01, },
    {(float)+5.71014829e-01, (float)+9.96650036e-01, (float)+8.45138754e-01, (float)+7.79611527e-01, (float)+4.81732294e-01, },
    {(float)+5.70319741e-01, (float)+9.96673794e-01, (float)+8.44362507e-01, (float)+7.79357623e-01, (float)+4.80873414e-01, },
    {(float)+5.69624652e-01, (float)+9.96673794e-01, (float)+8.43586260e-01, (float)+7.79103720e-01, (float)+4.80014535e-01, },
    {(float)+5.68929564e-01, (float)+9.96673794e-01, (float)+8.42810014e-01, (float)+7.78722864e-01, (float)+4.79155655e-01, },
    {(float)+5.68234476e-01, (float)+9.96697553e-01, (float)+8.41839705e-01, (float)+7.78468960e-01, (float)+4.78296776e-01, },
    {(float)+5.67539388e-01, (float)+9.96697553e-01, (float)+8.41063458e-01, (float)+7.78215056e-01, (float)+4.77437896e-01, },
    {(float)+5.66728452e-01, (float)+9.96697553e-01, (float)+8.40287211e-01, (float)+7.77834201e-01, (float)+4.76579017e-01, },
    {(float)+5.66033364e-01, (float)+9.96721311e-01, (float)+8.39510964e-01, (float)+7.77580297e-01, (float)+4.75703621e-01, },
    {(float)+5.65338276e-01, (float)+9.96721311e-01, (float)+8.38734718e-01, (float)+7.77199441e-01, (float)+4.74844741e-01, },
    {(float)+5.64643188e-01, (float)+9.96721311e-01, (float)+8.37764409e-01, (float)+7.76945538e-01, (float)+4.73985862e-01, },
    {(float)+5.63948100e-01, (float)+9.96745070e-01, (float)+8.36988162e-01, (float)+7.76691634e-01, (float)+4.73126982e-01, },
    {(float)+5.63253012e-01, (float)+9.96745070e-01, (float)+8.36211915e-01, (float)+7.76310778e-01, (float)+4.72268103e-01, },
    {(float)+5.62557924e-01, (float)+9.96745070e-01, (float)+8.35435669e-01, (float)+7.76056874e-01, (float)+4.71409223e-01, },
    {(float)+5.61862836e-01, (float)+9.96768829e-01, (float)+8.34659422e-01, (float)+7.75676019e-01, (float)+4.70550344e-01, },
    {(float)+5.61167748e-01, (float)+9.96768829e-01, (float)+8.33689113e-01, (float)+7.75422115e-01, (float)+4.69691464e-01, },
    {(float)+5.60472660e-01, (float)+9.96768829e-01, (float)+8.32912866e-01, (float)+7.75168211e-01, (float)+4.68816068e-01, },
    {(float)+5.59777572e-01, (float)+9.96792587e-01, (float)+8.32136619e-01, (float)+7.74787356e-01, (float)+4.67957188e-01, },
    {(float)+5.59082484e-01, (float)+9.96792587e-01, (float)+8.31360373e-01, (float)+7.74533452e-01, (float)+4.67098309e-01, },
    {(float)+5.58387396e-01, (float)+9.96792587e-01, (float)+8.30584126e-01, (float)+7.74279548e-01, (float)+4.66239429e-01, },
    {(float)+5.57692308e-01, (float)+9.96816346e-01, (float)+8.29613817e-01, (float)+7.73898692e-01, (float)+4.65380550e-01, },
    {(float)+5.56997220e-01, (float)+9.96816346e-01, (float)+8.28837570e-01, (float)+7.73644789e-01, (float)+4.64521670e-01, },
    {(float)+5.56302132e-01, (float)+9.96816346e-01, (float)+8.28061324e-01, (float)+7.73263933e-01, (float)+4.63662791e-01, },
    {(float)+5.55607044e-01, (float)+9.96840105e-01, (float)+8.27285077e-01, (float)+7.73010029e-01, (float)+4.62787394e-01, },
    {(float)+5.54911956e-01, (float)+9.96840105e-01, (float)+8.26508830e-01, (float)+7.72756125e-01, (float)+4.61928515e-01, },
    {(float)+5.54216867e-01, (float)+9.96840105e-01, (float)+8.25538521e-01, (float)+7.72375270e-01, (float)+4.61069635e-01, },
    {(float)+5.53521779e-01, (float)+9.96863863e-01, (float)+8.24762274e-01, (float)+7.72121366e-01, (float)+4.60210756e-01, },
    {(float)+5.52826691e-01, (float)+9.96863863e-01, (float)+8.23986028e-01, (float)+7.71740510e-01, (float)+4.59351876e-01, },
    {(float)+5.52131603e-01, (float)+9.96863863e-01, (float)+8.23209781e-01, (float)+7.71486607e-01, (float)+4.58492997e-01, },
    {(float)+5.51436515e-01, (float)+9.96887622e-01, (float)+8.22433534e-01, (float)+7.71232703e-01, (float)+4.57634117e-01, },
    {(float)+5.50741427e-01, (float)+9.96887622e-01, (float)+8.21463225e-01, (float)+7.70851847e-01, (float)+4.56758721e-01, },
    {(float)+5.50046339e-01, (float)+9.96887622e-01, (float)+8.20686978e-01, (float)+7.70597943e-01, (float)+4.55899841e-01, },
    {(float)+5.49351251e-01, (float)+9.96911380e-01, (float)+8.19910732e-01, (float)+7.70344040e-01, (float)+4.55040962e-01, },
    {(float)+5.48656163e-01, (float)+9.96911380e-01, (float)+8.19134485e-01, (float)+7.69963184e-01, (float)+4.54182082e-01, },
    {(float)+5.47961075e-01, (float)+9.96935139e-01, (float)+8.18358238e-01, (float)+7.69709280e-01, (float)+4.53323203e-01, },
    {(float)+5.47265987e-01, (float)+9.96935139e-01, (float)+8.17387929e-01, (float)+7.69328425e-01, (float)+4.52464323e-01, },
    {(float)+5.46570899e-01, (float)+9.96935139e-01, (float)+8.16611683e-01, (float)+7.69074521e-01, (float)+4.51605444e-01, },
    {(float)+5.45875811e-01, (float)+9.96958898e-01, (float)+8.15835436e-01, (float)+7.68820617e-01, (float)+4.50730048e-01, },
    {(float)+5.45180723e-01, (float)+9.96958898e-01, (float)+8.15059189e-01, (float)+7.68439761e-01, (float)+4.49871168e-01, },
    {(float)+5.44485635e-01, (float)+9.96958898e-01, (float)+8.14282942e-01, (float)+7.68185858e-01, (float)+4.49012289e-01, },
    {(float)+5.43790547e-01, (float)+9.96982656e-01, (float)+8.13312633e-01, (float)+7.67931954e-01, (float)+4.48153409e-01, },
    {(float)+5.43095459e-01, (float)+9.96982656e-01, (float)+8.12536387e-01, (float)+7.67551098e-01, (float)+4.47294530e-01, },
    {(float)+5.42400371e-01, (float)+9.96982656e-01, (float)+8.11760140e-01, (float)+7.67297194e-01, (float)+4.46435650e-01, },
    {(float)+5.41705283e-01, (float)+9.97006415e-01, (float)+8.10983893e-01, (float)+7.66916339e-01, (float)+4.45576771e-01, },
    {(float)+5.41010195e-01, (float)+9.97006415e-01, (float)+8.10207646e-01, (float)+7.66662435e-01, (float)+4.44701374e-01, },
    {(float)+5.40315107e-01, (float)+9.97006415e-01, (float)+8.09237337e-01, (float)+7.66408531e-01, (float)+4.43842495e-01, },
    {(float)+5.39620019e-01, (float)+9.97030173e-01, (float)+8.08461091e-01, (float)+7.66027676e-01, (float)+4.42983615e-01, },
    {(float)+5.38924930e-01, (float)+9.97030173e-01, (float)+8.07684844e-01, (float)+7.65773772e-01, (float)+4.42124736e-01, },
    {(float)+5.38229842e-01, (float)+9.97030173e-01, (float)+8.06908597e-01, (float)+7.65392916e-01, (float)+4.41265856e-01, },
    {(float)+5.37534754e-01, (float)+9.97053932e-01, (float)+8.06132350e-01, (float)+7.65139012e-01, (float)+4.40406977e-01, },
    {(float)+5.36839666e-01, (float)+9.97053932e-01, (float)+8.05162042e-01, (float)+7.64885109e-01, (float)+4.39548097e-01, },
    {(float)+5.36144578e-01, (float)+9.97053932e-01, (float)+8.04385795e-01, (float)+7.64504253e-01, (float)+4.38672701e-01, },
    {(float)+5.35449490e-01, (float)+9.97077691e-01, (float)+8.03609548e-01, (float)+7.64250349e-01, (float)+4.37813821e-01, },
    {(float)+5.34754402e-01, (float)+9.97077691e-01, (float)+8.02833301e-01, (float)+7.63996445e-01, (float)+4.36954942e-01, },
    {(float)+5.34059314e-01, (float)+9.97077691e-01, (float)+8.02057054e-01, (float)+7.63615590e-01, (float)+4.36096062e-01, },
    {(float)+5.33364226e-01, (float)+9.97101449e-01, (float)+8.01280807e-01, (float)+7.63361686e-01, (float)+4.35237183e-01, },
    {(float)+5.32669138e-01, (float)+9.97101449e-01, (float)+8.00310499e-01, (float)+7.62980830e-01, (float)+4.34378303e-01, },
    {(float)+5.31974050e-01, (float)+9.97101449e-01, (float)+7.99534252e-01, (float)+7.62726926e-01, (float)+4.33519424e-01, },
    {(float)+5.31278962e-01, (float)+9.97125208e-01, (float)+7.98758005e-01, (float)+7.62473023e-01, (float)+4.32644027e-01, },
    {(float)+5.30583874e-01, (float)+9.97125208e-01, (float)+7.97981758e-01, (float)+7.62092167e-01, (float)+4.31785148e-01, },
    {(float)+5.29888786e-01, (float)+9.97125208e-01, (float)+7.97205511e-01, (float)+7.61838263e-01, (float)+4.30926268e-01, },
    {(float)+5.29193698e-01, (float)+9.97148967e-01, (float)+7.96235203e-01, (float)+7.61457408e-01, (float)+4.30067389e-01, },
    {(float)+5.28498610e-01, (float)+9.97148967e-01, (float)+7.95458956e-01, (float)+7.61203504e-01, (float)+4.29208510e-01, },
    {(float)+5.27803522e-01, (float)+9.97148967e-01, (float)+7.94682709e-01, (float)+7.60949600e-01, (float)+4.28349630e-01, },
    {(float)+5.27108434e-01, (float)+9.97172725e-01, (float)+7.93906462e-01, (float)+7.60568744e-01, (float)+4.27490751e-01, },
    {(float)+5.26413346e-01, (float)+9.97172725e-01, (float)+7.93130215e-01, (float)+7.60314841e-01, (float)+4.26631871e-01, },
    {(float)+5.25718258e-01, (float)+9.97172725e-01, (float)+7.92159907e-01, (float)+7.60060937e-01, (float)+4.25756475e-01, },
    {(float)+5.25023170e-01, (float)+9.97196484e-01, (float)+7.91383660e-01, (float)+7.59680081e-01, (float)+4.24897595e-01, },
    {(float)+5.24328082e-01, (float)+9.97196484e-01, (float)+7.90607413e-01, (float)+7.59426177e-01, (float)+4.24038716e-01, },
    {(float)+5.23632994e-01, (float)+9.97196484e-01, (float)+7.89831166e-01, (float)+7.59045322e-01, (float)+4.23179836e-01, },
    {(float)+5.22937905e-01, (float)+9.97220242e-01, (float)+7.89054919e-01, (float)+7.58791418e-01, (float)+4.22320957e-01, },
    {(float)+5.22242817e-01, (float)+9.97220242e-01, (float)+7.88084611e-01, (float)+7.58537514e-01, (float)+4.21462077e-01, },
    {(float)+5.21547729e-01, (float)+9.97220242e-01, (float)+7.87308364e-01, (float)+7.58156659e-01, (float)+4.20603198e-01, },
    {(float)+5.20852641e-01, (float)+9.97244001e-01, (float)+7.86532117e-01, (float)+7.57902755e-01, (float)+4.19727801e-01, },
    {(float)+5.20157553e-01, (float)+9.97244001e-01, (float)+7.85755870e-01, (float)+7.57648851e-01, (float)+4.18868922e-01, },
    {(float)+5.19462465e-01, (float)+9.97244001e-01, (float)+7.84979624e-01, (float)+7.57267995e-01, (float)+4.18010042e-01, },
    {(float)+5.18767377e-01, (float)+9.97267760e-01, (float)+7.84009315e-01, (float)+7.57014092e-01, (float)+4.17151163e-01, },
    {(float)+5.18072289e-01, (float)+9.97267760e-01, (float)+7.83233068e-01, (float)+7.56633236e-01, (float)+4.16292283e-01, },
    {(float)+5.17377201e-01, (float)+9.97267760e-01, (float)+7.82456821e-01, (float)+7.56379332e-01, (float)+4.15433404e-01, },
    {(float)+5.16682113e-01, (float)+9.97291518e-01, (float)+7.81680574e-01, (float)+7.56125428e-01, (float)+4.14574524e-01, },
    {(float)+5.15987025e-01, (float)+9.97291518e-01, (float)+7.80904328e-01, (float)+7.55744573e-01, (float)+4.13699128e-01, },
    {(float)+5.15291937e-01, (float)+9.97291518e-01, (float)+7.79934019e-01, (float)+7.55490669e-01, (float)+4.12840248e-01, },
    {(float)+5.14481001e-01, (float)+9.97315277e-01, (float)+7.79157772e-01, (float)+7.55109813e-01, (float)+4.11981369e-01, },
    {(float)+5.13785913e-01, (float)+9.97315277e-01, (float)+7.78381525e-01, (float)+7.54855910e-01, (float)+4.11122489e-01, },
    {(float)+5.13090825e-01, (float)+9.97315277e-01, (float)+7.77605278e-01, (float)+7.54602006e-01, (float)+4.10263610e-01, },
    {(float)+5.12395737e-01, (float)+9.97339035e-01, (float)+7.76829032e-01, (float)+7.54221150e-01, (float)+4.09404730e-01, },
    {(float)+5.11700649e-01, (float)+9.97339035e-01, (float)+7.75858723e-01, (float)+7.53967246e-01, (float)+4.08545851e-01, },
    {(float)+5.11005561e-01, (float)+9.97339035e-01, (float)+7.75082476e-01, (float)+7.53713343e-01, (float)+4.07670455e-01, },
    {(float)+5.10310473e-01, (float)+9.97362794e-01, (float)+7.74306229e-01, (float)+7.53332487e-01, (float)+4.06811575e-01, },
    {(float)+5.09615385e-01, (float)+9.97362794e-01, (float)+7.73529983e-01, (float)+7.53078583e-01, (float)+4.05952696e-01, },
    {(float)+5.08920297e-01, (float)+9.97362794e-01, (float)+7.72753736e-01, (float)+7.52697728e-01, (float)+4.05093816e-01, },
    {(float)+5.08225209e-01, (float)+9.97386553e-01, (float)+7.71783427e-01, (float)+7.52443824e-01, (float)+4.04234937e-01, },
    {(float)+5.07530120e-01, (float)+9.97386553e-01, (float)+7.71007180e-01, (float)+7.52189920e-01, (float)+4.03376057e-01, },
    {(float)+5.06835032e-01, (float)+9.97386553e-01, (float)+7.70230933e-01, (float)+7.51809064e-01, (float)+4.02517178e-01, },
    {(float)+5.06139944e-01, (float)+9.97410311e-01, (float)+7.69454687e-01, (float)+7.51555161e-01, (float)+4.01641781e-01, },
    {(float)+5.05444856e-01, (float)+9.97410311e-01, (float)+7.68678440e-01, (float)+7.51174305e-01, (float)+4.00782902e-01, },
    {(float)+5.04749768e-01, (float)+9.97410311e-01, (float)+7.67708131e-01, (float)+7.50920401e-01, (float)+3.99924022e-01, },
    {(float)+5.04054680e-01, (float)+9.97434070e-01, (float)+7.66931884e-01, (float)+7.50666497e-01, (float)+3.99065143e-01, },
    {(float)+5.03359592e-01, (float)+9.97434070e-01, (float)+7.66155637e-01, (float)+7.50285642e-01, (float)+3.98206263e-01, },
    {(float)+5.02664504e-01, (float)+9.97434070e-01, (float)+7.65379391e-01, (float)+7.50031738e-01, (float)+3.97347384e-01, },
    {(float)+5.01969416e-01, (float)+9.97457828e-01, (float)+7.64603144e-01, (float)+7.49777834e-01, (float)+3.96488504e-01, },
    {(float)+5.01274328e-01, (float)+9.97457828e-01, (float)+7.63632835e-01, (float)+7.49396979e-01, (float)+3.95613108e-01, },
    {(float)+5.00579240e-01, (float)+9.97457828e-01, (float)+7.62856588e-01, (float)+7.49143075e-01, (float)+3.94754228e-01, },
    {(float)+4.99884152e-01, (float)+9.97481587e-01, (float)+7.62080342e-01, (float)+7.48762219e-01, (float)+3.93895349e-01, },
    {(float)+4.99189064e-01, (float)+9.97481587e-01, (float)+7.61304095e-01, (float)+7.48508315e-01, (float)+3.93036469e-01, },
    {(float)+4.98493976e-01, (float)+9.97481587e-01, (float)+7.60527848e-01, (float)+7.48254412e-01, (float)+3.92177590e-01, },
    {(float)+4.97798888e-01, (float)+9.97505346e-01, (float)+7.59557539e-01, (float)+7.47873556e-01, (float)+3.91318710e-01, },
    {(float)+4.97103800e-01, (float)+9.97505346e-01, (float)+7.58781292e-01, (float)+7.47619652e-01, (float)+3.90459831e-01, },
    {(float)+4.96408712e-01, (float)+9.97505346e-01, (float)+7.58005046e-01, (float)+7.47365748e-01, (float)+3.89584434e-01, },
    {(float)+4.95713624e-01, (float)+9.97529104e-01, (float)+7.57228799e-01, (float)+7.46984893e-01, (float)+3.88725555e-01, },
    {(float)+4.95018536e-01, (float)+9.97529104e-01, (float)+7.56452552e-01, (float)+7.46730989e-01, (float)+3.87866675e-01, },
    {(float)+4.94323448e-01, (float)+9.97529104e-01, (float)+7.55482243e-01, (float)+7.46350133e-01, (float)+3.87007796e-01, },
    {(float)+4.93628360e-01, (float)+9.97552863e-01, (float)+7.54705997e-01, (float)+7.46096230e-01, (float)+3.86148916e-01, },
    {(float)+4.92933272e-01, (float)+9.97552863e-01, (float)+7.53929750e-01, (float)+7.45842326e-01, (float)+3.85290037e-01, },
    {(float)+4.92238184e-01, (float)+9.97552863e-01, (float)+7.53153503e-01, (float)+7.45461470e-01, (float)+3.84431158e-01, },
    {(float)+4.91543095e-01, (float)+9.97576622e-01, (float)+7.52377256e-01, (float)+7.45207566e-01, (float)+3.83555761e-01, },
    {(float)+4.90848007e-01, (float)+9.97576622e-01, (float)+7.51406947e-01, (float)+7.44826711e-01, (float)+3.82696882e-01, },
    {(float)+4.90152919e-01, (float)+9.97576622e-01, (float)+7.50630701e-01, (float)+7.44572807e-01, (float)+3.81838002e-01, },
    {(float)+4.89457831e-01, (float)+9.97600380e-01, (float)+7.49854454e-01, (float)+7.44318903e-01, (float)+3.80979123e-01, },
    {(float)+4.88762743e-01, (float)+9.97600380e-01, (float)+7.49078207e-01, (float)+7.43938047e-01, (float)+3.80120243e-01, },
    {(float)+4.88067655e-01, (float)+9.97624139e-01, (float)+7.48301960e-01, (float)+7.43684144e-01, (float)+3.79261364e-01, },
    {(float)+4.87372567e-01, (float)+9.97624139e-01, (float)+7.47525713e-01, (float)+7.43430240e-01, (float)+3.78402484e-01, },
    {(float)+4.86677479e-01, (float)+9.97624139e-01, (float)+7.46555405e-01, (float)+7.43049384e-01, (float)+3.77543605e-01, },
    {(float)+4.85982391e-01, (float)+9.97647897e-01, (float)+7.45779158e-01, (float)+7.42795481e-01, (float)+3.76668208e-01, },
    {(float)+4.85287303e-01, (float)+9.97647897e-01, (float)+7.45002911e-01, (float)+7.42414625e-01, (float)+3.75809329e-01, },
    {(float)+4.84592215e-01, (float)+9.97647897e-01, (float)+7.44226664e-01, (float)+7.42160721e-01, (float)+3.74950449e-01, },
    {(float)+4.83897127e-01, (float)+9.97671656e-01, (float)+7.43450417e-01, (float)+7.41906817e-01, (float)+3.74091570e-01, },
    {(float)+4.83202039e-01, (float)+9.97671656e-01, (float)+7.42480109e-01, (float)+7.41525962e-01, (float)+3.73232690e-01, },
    {(float)+4.82506951e-01, (float)+9.97671656e-01, (float)+7.41703862e-01, (float)+7.41272058e-01, (float)+3.72373811e-01, },
    {(float)+4.81811863e-01, (float)+9.97695415e-01, (float)+7.40927615e-01, (float)+7.40891202e-01, (float)+3.71514931e-01, },
    {(float)+4.81116775e-01, (float)+9.97695415e-01, (float)+7.40151368e-01, (float)+7.40637298e-01, (float)+3.70639535e-01, },
    {(float)+4.80421687e-01, (float)+9.97695415e-01, (float)+7.39375121e-01, (float)+7.40383395e-01, (float)+3.69780655e-01, },
    {(float)+4.79726599e-01, (float)+9.97719173e-01, (float)+7.38404813e-01, (float)+7.40002539e-01, (float)+3.68921776e-01, },
    {(float)+4.79031511e-01, (float)+9.97719173e-01, (float)+7.37628566e-01, (float)+7.39748635e-01, (float)+3.68062896e-01, },
    {(float)+4.78336423e-01, (float)+9.97719173e-01, (float)+7.36852319e-01, (float)+7.39494731e-01, (float)+3.67204017e-01, },
    {(float)+4.77641335e-01, (float)+9.97742932e-01, (float)+7.36076072e-01, (float)+7.39113876e-01, (float)+3.66345137e-01, },
    {(float)+4.76946247e-01, (float)+9.97742932e-01, (float)+7.35299825e-01, (float)+7.38859972e-01, (float)+3.65486258e-01, },
    {(float)+4.76251158e-01, (float)+9.97742932e-01, (float)+7.34329517e-01, (float)+7.38479116e-01, (float)+3.64610862e-01, },
    {(float)+4.75556070e-01, (float)+9.97766690e-01, (float)+7.33553270e-01, (float)+7.38225213e-01, (float)+3.63751982e-01, },
    {(float)+4.74860982e-01, (float)+9.97766690e-01, (float)+7.32777023e-01, (float)+7.37971309e-01, (float)+3.62893103e-01, },
    {(float)+4.74165894e-01, (float)+9.97766690e-01, (float)+7.32000776e-01, (float)+7.37590453e-01, (float)+3.62034223e-01, },
    {(float)+4.73470806e-01, (float)+9.97790449e-01, (float)+7.31224529e-01, (float)+7.37336549e-01, (float)+3.61175344e-01, },
    {(float)+4.72775718e-01, (float)+9.97790449e-01, (float)+7.30254221e-01, (float)+7.37082646e-01, (float)+3.60316464e-01, },
    {(float)+4.72080630e-01, (float)+9.97790449e-01, (float)+7.29477974e-01, (float)+7.36701790e-01, (float)+3.59457585e-01, },
    {(float)+4.71385542e-01, (float)+9.97814208e-01, (float)+7.28701727e-01, (float)+7.36447886e-01, (float)+3.58582188e-01, },
    {(float)+4.70690454e-01, (float)+9.97814208e-01, (float)+7.27925480e-01, (float)+7.36067031e-01, (float)+3.57723309e-01, },
    {(float)+4.69995366e-01, (float)+9.97814208e-01, (float)+7.27149233e-01, (float)+7.35813127e-01, (float)+3.56864429e-01, },
    {(float)+4.69300278e-01, (float)+9.97837966e-01, (float)+7.26178925e-01, (float)+7.35559223e-01, (float)+3.56005550e-01, },
    {(float)+4.68605190e-01, (float)+9.97837966e-01, (float)+7.25402678e-01, (float)+7.35178367e-01, (float)+3.55146670e-01, },
    {(float)+4.67910102e-01, (float)+9.97837966e-01, (float)+7.24626431e-01, (float)+7.34924464e-01, (float)+3.54287791e-01, },
    {(float)+4.67215014e-01, (float)+9.97861725e-01, (float)+7.23850184e-01, (float)+7.34543608e-01, (float)+3.53428911e-01, },
    {(float)+4.66519926e-01, (float)+9.97861725e-01, (float)+7.23073938e-01, (float)+7.34289704e-01, (float)+3.52553515e-01, },
    {(float)+4.65824838e-01, (float)+9.97861725e-01, (float)+7.22103629e-01, (float)+7.34035800e-01, (float)+3.51694635e-01, },
    {(float)+4.65129750e-01, (float)+9.97885483e-01, (float)+7.21327382e-01, (float)+7.33654945e-01, (float)+3.50835756e-01, },
    {(float)+4.64434662e-01, (float)+9.97885483e-01, (float)+7.20551135e-01, (float)+7.33401041e-01, (float)+3.49976876e-01, },
    {(float)+4.63739574e-01, (float)+9.97885483e-01, (float)+7.19774888e-01, (float)+7.33147137e-01, (float)+3.49117997e-01, },
    {(float)+4.62928638e-01, (float)+9.97909242e-01, (float)+7.18998642e-01, (float)+7.32766282e-01, (float)+3.48259117e-01, },
    {(float)+4.62233550e-01, (float)+9.97909242e-01, (float)+7.18028333e-01, (float)+7.32512378e-01, (float)+3.47400238e-01, },
    {(float)+4.61538462e-01, (float)+9.97909242e-01, (float)+7.17252086e-01, (float)+7.32131522e-01, (float)+3.46524841e-01, },
    {(float)+4.60843373e-01, (float)+9.97933001e-01, (float)+7.16475839e-01, (float)+7.31877618e-01, (float)+3.45665962e-01, },
    {(float)+4.60148285e-01, (float)+9.97933001e-01, (float)+7.15699592e-01, (float)+7.31623715e-01, (float)+3.44807082e-01, },
    {(float)+4.59453197e-01, (float)+9.97933001e-01, (float)+7.14923346e-01, (float)+7.31242859e-01, (float)+3.43948203e-01, },
    {(float)+4.58758109e-01, (float)+9.97956759e-01, (float)+7.13953037e-01, (float)+7.30988955e-01, (float)+3.43089323e-01, },
    {(float)+4.58063021e-01, (float)+9.97956759e-01, (float)+7.13176790e-01, (float)+7.30608100e-01, (float)+3.42230444e-01, },
    {(float)+4.57367933e-01, (float)+9.97956759e-01, (float)+7.12400543e-01, (float)+7.30354196e-01, (float)+3.41371564e-01, },
    {(float)+4.56672845e-01, (float)+9.97980518e-01, (float)+7.11624297e-01, (float)+7.30100292e-01, (float)+3.40496168e-01, },
    {(float)+4.55977757e-01, (float)+9.97980518e-01, (float)+7.10848050e-01, (float)+7.29719436e-01, (float)+3.39637289e-01, },
    {(float)+4.55282669e-01, (float)+9.97980518e-01, (float)+7.09877741e-01, (float)+7.29465533e-01, (float)+3.38778409e-01, },
    {(float)+4.54587581e-01, (float)+9.98004277e-01, (float)+7.09101494e-01, (float)+7.29211629e-01, (float)+3.37919530e-01, },
    {(float)+4.53892493e-01, (float)+9.98004277e-01, (float)+7.08325247e-01, (float)+7.28830773e-01, (float)+3.37060650e-01, },
    {(float)+4.53197405e-01, (float)+9.98004277e-01, (float)+7.07549001e-01, (float)+7.28576869e-01, (float)+3.36201771e-01, },
    {(float)+4.52502317e-01, (float)+9.98028035e-01, (float)+7.06772754e-01, (float)+7.28196014e-01, (float)+3.35342891e-01, },
    {(float)+4.51807229e-01, (float)+9.98028035e-01, (float)+7.05802445e-01, (float)+7.27942110e-01, (float)+3.34467495e-01, },
    {(float)+4.51112141e-01, (float)+9.98028035e-01, (float)+7.05026198e-01, (float)+7.27688206e-01, (float)+3.33608615e-01, },
    {(float)+4.50417053e-01, (float)+9.98051794e-01, (float)+7.04249951e-01, (float)+7.27307351e-01, (float)+3.32749736e-01, },
    {(float)+4.49721965e-01, (float)+9.98051794e-01, (float)+7.03473705e-01, (float)+7.27053447e-01, (float)+3.31890856e-01, },
    {(float)+4.49026877e-01, (float)+9.98051794e-01, (float)+7.02697458e-01, (float)+7.26799543e-01, (float)+3.31031977e-01, },
    {(float)+4.48331789e-01, (float)+9.98075552e-01, (float)+7.01727149e-01, (float)+7.26418687e-01, (float)+3.30173097e-01, },
    {(float)+4.47636701e-01, (float)+9.98075552e-01, (float)+7.00950902e-01, (float)+7.26164784e-01, (float)+3.29314218e-01, },
    {(float)+4.46941613e-01, (float)+9.98075552e-01, (float)+7.00174656e-01, (float)+7.25783928e-01, (float)+3.28455338e-01, },
    {(float)+4.46246525e-01, (float)+9.98099311e-01, (float)+6.99398409e-01, (float)+7.25530024e-01, (float)+3.27579942e-01, },
    {(float)+4.45551437e-01, (float)+9.98099311e-01, (float)+6.98622162e-01, (float)+7.25276120e-01, (float)+3.26721062e-01, },
    {(float)+4.44856348e-01, (float)+9.98099311e-01, (float)+6.97651853e-01, (float)+7.24895265e-01, (float)+3.25862183e-01, },
    {(float)+4.44161260e-01, (float)+9.98123070e-01, (float)+6.96875606e-01, (float)+7.24641361e-01, (float)+3.25003303e-01, },
    {(float)+4.43466172e-01, (float)+9.98123070e-01, (float)+6.96099360e-01, (float)+7.24260505e-01, (float)+3.24144424e-01, },
    {(float)+4.42771084e-01, (float)+9.98123070e-01, (float)+6.95323113e-01, (float)+7.24006601e-01, (float)+3.23285544e-01, },
    {(float)+4.42075996e-01, (float)+9.98146828e-01, (float)+6.94546866e-01, (float)+7.23752698e-01, (float)+3.22426665e-01, },
    {(float)+4.41380908e-01, (float)+9.98146828e-01, (float)+6.93770619e-01, (float)+7.23371842e-01, (float)+3.21551268e-01, },
    {(float)+4.40685820e-01, (float)+9.98146828e-01, (float)+6.92800310e-01, (float)+7.23117938e-01, (float)+3.20692389e-01, },
    {(float)+4.39990732e-01, (float)+9.98170587e-01, (float)+6.92024064e-01, (float)+7.22864035e-01, (float)+3.19833510e-01, },
    {(float)+4.39295644e-01, (float)+9.98170587e-01, (float)+6.91247817e-01, (float)+7.22483179e-01, (float)+3.18974630e-01, },
    {(float)+4.38600556e-01, (float)+9.98170587e-01, (float)+6.90471570e-01, (float)+7.22229275e-01, (float)+3.18115751e-01, },
    {(float)+4.37905468e-01, (float)+9.98194345e-01, (float)+6.89695323e-01, (float)+7.21848419e-01, (float)+3.17256871e-01, },
    {(float)+4.37210380e-01, (float)+9.98194345e-01, (float)+6.88725015e-01, (float)+7.21594516e-01, (float)+3.16397992e-01, },
    {(float)+4.36515292e-01, (float)+9.98194345e-01, (float)+6.87948768e-01, (float)+7.21340612e-01, (float)+3.15522595e-01, },
    {(float)+4.35820204e-01, (float)+9.98218104e-01, (float)+6.87172521e-01, (float)+7.20959756e-01, (float)+3.14663716e-01, },
    {(float)+4.35125116e-01, (float)+9.98218104e-01, (float)+6.86396274e-01, (float)+7.20705852e-01, (float)+3.13804836e-01, },
    {(float)+4.34430028e-01, (float)+9.98218104e-01, (float)+6.85620027e-01, (float)+7.20324997e-01, (float)+3.12945957e-01, },
    {(float)+4.33734940e-01, (float)+9.98241863e-01, (float)+6.84649719e-01, (float)+7.20071093e-01, (float)+3.12087077e-01, },
    {(float)+4.33039852e-01, (float)+9.98241863e-01, (float)+6.83873472e-01, (float)+7.19817189e-01, (float)+3.11228198e-01, },
    {(float)+4.32344764e-01, (float)+9.98241863e-01, (float)+6.83097225e-01, (float)+7.19436334e-01, (float)+3.10369318e-01, },
    {(float)+4.31649676e-01, (float)+9.98265621e-01, (float)+6.82320978e-01, (float)+7.19182430e-01, (float)+3.09493922e-01, },
    {(float)+4.30954588e-01, (float)+9.98265621e-01, (float)+6.81544731e-01, (float)+7.18928526e-01, (float)+3.08635042e-01, },
    {(float)+4.30259500e-01, (float)+9.98265621e-01, (float)+6.80574423e-01, (float)+7.18547670e-01, (float)+3.07776163e-01, },
    {(float)+4.29564411e-01, (float)+9.98289380e-01, (float)+6.79798176e-01, (float)+7.18293767e-01, (float)+3.06917283e-01, },
    {(float)+4.28869323e-01, (float)+9.98289380e-01, (float)+6.79021929e-01, (float)+7.17912911e-01, (float)+3.06058404e-01, },
    {(float)+4.28174235e-01, (float)+9.98313139e-01, (float)+6.78245682e-01, (float)+7.17659007e-01, (float)+3.05199524e-01, },
    {(float)+4.27479147e-01, (float)+9.98313139e-01, (float)+6.77469435e-01, (float)+7.17405103e-01, (float)+3.04340645e-01, },
    {(float)+4.26784059e-01, (float)+9.98313139e-01, (float)+6.76499127e-01, (float)+7.17024248e-01, (float)+3.03465248e-01, },
    {(float)+4.26088971e-01, (float)+9.98336897e-01, (float)+6.75722880e-01, (float)+7.16770344e-01, (float)+3.02606369e-01, },
    {(float)+4.25393883e-01, (float)+9.98336897e-01, (float)+6.74946633e-01, (float)+7.16516440e-01, (float)+3.01747489e-01, },
    {(float)+4.24698795e-01, (float)+9.98336897e-01, (float)+6.74170386e-01, (float)+7.16135585e-01, (float)+3.00888610e-01, },
    {(float)+4.24003707e-01, (float)+9.98360656e-01, (float)+6.73394139e-01, (float)+7.15881681e-01, (float)+3.00029730e-01, },
    {(float)+4.23308619e-01, (float)+9.98360656e-01, (float)+6.72423831e-01, (float)+7.15500825e-01, (float)+2.99170851e-01, },
    {(float)+4.22613531e-01, (float)+9.98360656e-01, (float)+6.71647584e-01, (float)+7.15246921e-01, (float)+2.98311971e-01, },
    {(float)+4.21918443e-01, (float)+9.98384414e-01, (float)+6.70871337e-01, (float)+7.14993018e-01, (float)+2.97436575e-01, },
    {(float)+4.21223355e-01, (float)+9.98384414e-01, (float)+6.70095090e-01, (float)+7.14612162e-01, (float)+2.96577696e-01, },
    {(float)+4.20528267e-01, (float)+9.98384414e-01, (float)+6.69318843e-01, (float)+7.14358258e-01, (float)+2.95718816e-01, },
    {(float)+4.19833179e-01, (float)+9.98408173e-01, (float)+6.68348535e-01, (float)+7.13977403e-01, (float)+2.94859937e-01, },
    {(float)+4.19138091e-01, (float)+9.98408173e-01, (float)+6.67572288e-01, (float)+7.13723499e-01, (float)+2.94001057e-01, },
    {(float)+4.18443003e-01, (float)+9.98408173e-01, (float)+6.66796041e-01, (float)+7.13469595e-01, (float)+2.93142178e-01, },
    {(float)+4.17747915e-01, (float)+9.98431932e-01, (float)+6.66019794e-01, (float)+7.13088739e-01, (float)+2.92283298e-01, },
    {(float)+4.17052827e-01, (float)+9.98431932e-01, (float)+6.65243547e-01, (float)+7.12834836e-01, (float)+2.91407902e-01, },
    {(float)+4.16357739e-01, (float)+9.98431932e-01, (float)+6.64273239e-01, (float)+7.12580932e-01, (float)+2.90549022e-01, },
    {(float)+4.15662651e-01, (float)+9.98455690e-01, (float)+6.63496992e-01, (float)+7.12200076e-01, (float)+2.89690143e-01, },
    {(float)+4.14967563e-01, (float)+9.98455690e-01, (float)+6.62720745e-01, (float)+7.11946172e-01, (float)+2.88831263e-01, },
    {(float)+4.14272475e-01, (float)+9.98455690e-01, (float)+6.61944498e-01, (float)+7.11565317e-01, (float)+2.87972384e-01, },
    {(float)+4.13577386e-01, (float)+9.98479449e-01, (float)+6.61168252e-01, (float)+7.11311413e-01, (float)+2.87113504e-01, },
    {(float)+4.12882298e-01, (float)+9.98479449e-01, (float)+6.60197943e-01, (float)+7.11057509e-01, (float)+2.86254625e-01, },
    {(float)+4.12187210e-01, (float)+9.98479449e-01, (float)+6.59421696e-01, (float)+7.10676654e-01, (float)+2.85395745e-01, },
    {(float)+4.11492122e-01, (float)+9.98503207e-01, (float)+6.58645449e-01, (float)+7.10422750e-01, (float)+2.84520349e-01, },
    {(float)+4.10681186e-01, (float)+9.98503207e-01, (float)+6.57869202e-01, (float)+7.10041894e-01, (float)+2.83661469e-01, },
    {(float)+4.09986098e-01, (float)+9.98503207e-01, (float)+6.57092956e-01, (float)+7.09787990e-01, (float)+2.82802590e-01, },
    {(float)+4.09291010e-01, (float)+9.98526966e-01, (float)+6.56122647e-01, (float)+7.09534087e-01, (float)+2.81943710e-01, },
    {(float)+4.08595922e-01, (float)+9.98526966e-01, (float)+6.55346400e-01, (float)+7.09153231e-01, (float)+2.81084831e-01, },
    {(float)+4.07900834e-01, (float)+9.98526966e-01, (float)+6.54570153e-01, (float)+7.08899327e-01, (float)+2.80225951e-01, },
    {(float)+4.07205746e-01, (float)+9.98550725e-01, (float)+6.53793906e-01, (float)+7.08645423e-01, (float)+2.79367072e-01, },
    {(float)+4.06510658e-01, (float)+9.98550725e-01, (float)+6.53017660e-01, (float)+7.08264568e-01, (float)+2.78491675e-01, },
    {(float)+4.05815570e-01, (float)+9.98550725e-01, (float)+6.52047351e-01, (float)+7.08010664e-01, (float)+2.77632796e-01, },
    {(float)+4.05120482e-01, (float)+9.98574483e-01, (float)+6.51271104e-01, (float)+7.07629808e-01, (float)+2.76773916e-01, },
    {(float)+4.04425394e-01, (float)+9.98574483e-01, (float)+6.50494857e-01, (float)+7.07375905e-01, (float)+2.75915037e-01, },
    {(float)+4.03730306e-01, (float)+9.98574483e-01, (float)+6.49718611e-01, (float)+7.07122001e-01, (float)+2.75056158e-01, },
    {(float)+4.03035218e-01, (float)+9.98598242e-01, (float)+6.48942364e-01, (float)+7.06741145e-01, (float)+2.74197278e-01, },
    {(float)+4.02340130e-01, (float)+9.98598242e-01, (float)+6.47972055e-01, (float)+7.06487241e-01, (float)+2.73338399e-01, },
    {(float)+4.01645042e-01, (float)+9.98598242e-01, (float)+6.47195808e-01, (float)+7.06233338e-01, (float)+2.72463002e-01, },
    {(float)+4.00949954e-01, (float)+9.98622000e-01, (float)+6.46419561e-01, (float)+7.05852482e-01, (float)+2.71604123e-01, },
    {(float)+4.00254866e-01, (float)+9.98622000e-01, (float)+6.45643315e-01, (float)+7.05598578e-01, (float)+2.70745243e-01, },
    {(float)+3.99559778e-01, (float)+9.98622000e-01, (float)+6.44867068e-01, (float)+7.05217722e-01, (float)+2.69886364e-01, },
    {(float)+3.98864690e-01, (float)+9.98645759e-01, (float)+6.43896759e-01, (float)+7.04963819e-01, (float)+2.69027484e-01, },
    {(float)+3.98169601e-01, (float)+9.98645759e-01, (float)+6.43120512e-01, (float)+7.04709915e-01, (float)+2.68168605e-01, },
    {(float)+3.97474513e-01, (float)+9.98645759e-01, (float)+6.42344265e-01, (float)+7.04329059e-01, (float)+2.67309725e-01, },
    {(float)+3.96779425e-01, (float)+9.98669518e-01, (float)+6.41568019e-01, (float)+7.04075156e-01, (float)+2.66434329e-01, },
    {(float)+3.96084337e-01, (float)+9.98669518e-01, (float)+6.40791772e-01, (float)+7.03694300e-01, (float)+2.65575449e-01, },
    {(float)+3.95389249e-01, (float)+9.98669518e-01, (float)+6.40015525e-01, (float)+7.03440396e-01, (float)+2.64716570e-01, },
    {(float)+3.94694161e-01, (float)+9.98693276e-01, (float)+6.39045216e-01, (float)+7.03186492e-01, (float)+2.63857690e-01, },
    {(float)+3.93999073e-01, (float)+9.98693276e-01, (float)+6.38268970e-01, (float)+7.02805637e-01, (float)+2.62998811e-01, },
    {(float)+3.93303985e-01, (float)+9.98693276e-01, (float)+6.37492723e-01, (float)+7.02551733e-01, (float)+2.62139931e-01, },
    {(float)+3.92608897e-01, (float)+9.98717035e-01, (float)+6.36716476e-01, (float)+7.02297829e-01, (float)+2.61281052e-01, },
    {(float)+3.91913809e-01, (float)+9.98717035e-01, (float)+6.35940229e-01, (float)+7.01916973e-01, (float)+2.60405655e-01, },
    {(float)+3.91218721e-01, (float)+9.98717035e-01, (float)+6.34969920e-01, (float)+7.01663070e-01, (float)+2.59546776e-01, },
    {(float)+3.90523633e-01, (float)+9.98740794e-01, (float)+6.34193674e-01, (float)+7.01282214e-01, (float)+2.58687896e-01, },
    {(float)+3.89828545e-01, (float)+9.98740794e-01, (float)+6.33417427e-01, (float)+7.01028310e-01, (float)+2.57829017e-01, },
    {(float)+3.89133457e-01, (float)+9.98740794e-01, (float)+6.32641180e-01, (float)+7.00774406e-01, (float)+2.56970137e-01, },
    {(float)+3.88438369e-01, (float)+9.98764552e-01, (float)+6.31864933e-01, (float)+7.00393551e-01, (float)+2.56111258e-01, },
    {(float)+3.87743281e-01, (float)+9.98764552e-01, (float)+6.30894624e-01, (float)+7.00139647e-01, (float)+2.55252378e-01, },
    {(float)+3.87048193e-01, (float)+9.98764552e-01, (float)+6.30118378e-01, (float)+6.99758791e-01, (float)+2.54376982e-01, },
    {(float)+3.86353105e-01, (float)+9.98788311e-01, (float)+6.29342131e-01, (float)+6.99504888e-01, (float)+2.53518103e-01, },
    {(float)+3.85658017e-01, (float)+9.98788311e-01, (float)+6.28565884e-01, (float)+6.99250984e-01, (float)+2.52659223e-01, },
    {(float)+3.84962929e-01, (float)+9.98788311e-01, (float)+6.27789637e-01, (float)+6.98870128e-01, (float)+2.51800344e-01, },
    {(float)+3.84267841e-01, (float)+9.98812069e-01, (float)+6.26819329e-01, (float)+6.98616224e-01, (float)+2.50941464e-01, },
    {(float)+3.83572753e-01, (float)+9.98812069e-01, (float)+6.26043082e-01, (float)+6.98362321e-01, (float)+2.50082585e-01, },
    {(float)+3.82877665e-01, (float)+9.98812069e-01, (float)+6.25266835e-01, (float)+6.97981465e-01, (float)+2.49223705e-01, },
    {(float)+3.82182576e-01, (float)+9.98835828e-01, (float)+6.24490588e-01, (float)+6.97727561e-01, (float)+2.48348309e-01, },
    {(float)+3.81487488e-01, (float)+9.98835828e-01, (float)+6.23714341e-01, (float)+6.97346706e-01, (float)+2.47489429e-01, },
    {(float)+3.80792400e-01, (float)+9.98835828e-01, (float)+6.22744033e-01, (float)+6.97092802e-01, (float)+2.46630550e-01, },
    {(float)+3.80097312e-01, (float)+9.98859587e-01, (float)+6.21967786e-01, (float)+6.96838898e-01, (float)+2.45771670e-01, },
    {(float)+3.79402224e-01, (float)+9.98859587e-01, (float)+6.21191539e-01, (float)+6.96458042e-01, (float)+2.44912791e-01, },
    {(float)+3.78707136e-01, (float)+9.98859587e-01, (float)+6.20415292e-01, (float)+6.96204139e-01, (float)+2.44053911e-01, },
    {(float)+3.78012048e-01, (float)+9.98883345e-01, (float)+6.19639045e-01, (float)+6.95950235e-01, (float)+2.43195032e-01, },
    {(float)+3.77316960e-01, (float)+9.98883345e-01, (float)+6.18668737e-01, (float)+6.95569379e-01, (float)+2.42319635e-01, },
    {(float)+3.76621872e-01, (float)+9.98883345e-01, (float)+6.17892490e-01, (float)+6.95315475e-01, (float)+2.41460756e-01, },
    {(float)+3.75926784e-01, (float)+9.98907104e-01, (float)+6.17116243e-01, (float)+6.94934620e-01, (float)+2.40601876e-01, },
    {(float)+3.75231696e-01, (float)+9.98907104e-01, (float)+6.16339996e-01, (float)+6.94680716e-01, (float)+2.39742997e-01, },
    {(float)+3.74536608e-01, (float)+9.98907104e-01, (float)+6.15563749e-01, (float)+6.94426812e-01, (float)+2.38884117e-01, },
    {(float)+3.73841520e-01, (float)+9.98930862e-01, (float)+6.14593441e-01, (float)+6.94045957e-01, (float)+2.38025238e-01, },
    {(float)+3.73146432e-01, (float)+9.98930862e-01, (float)+6.13817194e-01, (float)+6.93792053e-01, (float)+2.37166358e-01, },
    {(float)+3.72451344e-01, (float)+9.98930862e-01, (float)+6.13040947e-01, (float)+6.93411197e-01, (float)+2.36307479e-01, },
    {(float)+3.71756256e-01, (float)+9.98954621e-01, (float)+6.12264700e-01, (float)+6.93157293e-01, (float)+2.35432082e-01, },
    {(float)+3.71061168e-01, (float)+9.98954621e-01, (float)+6.11488453e-01, (float)+6.92903390e-01, (float)+2.34573203e-01, },
    {(float)+3.70366080e-01, (float)+9.98954621e-01, (float)+6.10518145e-01, (float)+6.92522534e-01, (float)+2.33714323e-01, },
    {(float)+3.69670992e-01, (float)+9.98978380e-01, (float)+6.09741898e-01, (float)+6.92268630e-01, (float)+2.32855444e-01, },
    {(float)+3.68975904e-01, (float)+9.98978380e-01, (float)+6.08965651e-01, (float)+6.92014726e-01, (float)+2.31996564e-01, },
    {(float)+3.68280816e-01, (float)+9.99002138e-01, (float)+6.08189404e-01, (float)+6.91633871e-01, (float)+2.31137685e-01, },
    {(float)+3.67585728e-01, (float)+9.99002138e-01, (float)+6.07413157e-01, (float)+6.91379967e-01, (float)+2.30278805e-01, },
    {(float)+3.66890639e-01, (float)+9.99002138e-01, (float)+6.06442849e-01, (float)+6.90999111e-01, (float)+2.29403409e-01, },
    {(float)+3.66195551e-01, (float)+9.99025897e-01, (float)+6.05666602e-01, (float)+6.90745208e-01, (float)+2.28544530e-01, },
    {(float)+3.65500463e-01, (float)+9.99025897e-01, (float)+6.04890355e-01, (float)+6.90491304e-01, (float)+2.27685650e-01, },
    {(float)+3.64805375e-01, (float)+9.99025897e-01, (float)+6.04114108e-01, (float)+6.90110448e-01, (float)+2.26826771e-01, },
    {(float)+3.64110287e-01, (float)+9.99049656e-01, (float)+6.03337861e-01, (float)+6.89856544e-01, (float)+2.25967891e-01, },
    {(float)+3.63415199e-01, (float)+9.99049656e-01, (float)+6.02367553e-01, (float)+6.89475689e-01, (float)+2.25109012e-01, },
    {(float)+3.62720111e-01, (float)+9.99049656e-01, (float)+6.01591306e-01, (float)+6.89221785e-01, (float)+2.24250132e-01, },
    {(float)+3.62025023e-01, (float)+9.99073414e-01, (float)+6.00815059e-01, (float)+6.88967881e-01, (float)+2.23374736e-01, },
    {(float)+3.61329935e-01, (float)+9.99073414e-01, (float)+6.00038812e-01, (float)+6.88587026e-01, (float)+2.22515856e-01, },
    {(float)+3.60634847e-01, (float)+9.99073414e-01, (float)+5.99262565e-01, (float)+6.88333122e-01, (float)+2.21656977e-01, },
    {(float)+3.59939759e-01, (float)+9.99097173e-01, (float)+5.98292257e-01, (float)+6.88079218e-01, (float)+2.20798097e-01, },
    {(float)+3.59244671e-01, (float)+9.99097173e-01, (float)+5.97516010e-01, (float)+6.87698362e-01, (float)+2.19939218e-01, },
    {(float)+3.58433735e-01, (float)+9.99097173e-01, (float)+5.96739763e-01, (float)+6.87444459e-01, (float)+2.19080338e-01, },
    {(float)+3.57738647e-01, (float)+9.99120931e-01, (float)+5.95963516e-01, (float)+6.87063603e-01, (float)+2.18221459e-01, },
    {(float)+3.57043559e-01, (float)+9.99120931e-01, (float)+5.95187270e-01, (float)+6.86809699e-01, (float)+2.17346062e-01, },
    {(float)+3.56348471e-01, (float)+9.99120931e-01, (float)+5.94216961e-01, (float)+6.86555795e-01, (float)+2.16487183e-01, },
    {(float)+3.55653383e-01, (float)+9.99144690e-01, (float)+5.93440714e-01, (float)+6.86174940e-01, (float)+2.15628303e-01, },
    {(float)+3.54958295e-01, (float)+9.99144690e-01, (float)+5.92664467e-01, (float)+6.85921036e-01, (float)+2.14769424e-01, },
    {(float)+3.54263207e-01, (float)+9.99144690e-01, (float)+5.91888220e-01, (float)+6.85667132e-01, (float)+2.13910544e-01, },
    {(float)+3.53568119e-01, (float)+9.99168449e-01, (float)+5.91111974e-01, (float)+6.85286277e-01, (float)+2.13051665e-01, },
    {(float)+3.52873031e-01, (float)+9.99168449e-01, (float)+5.90141665e-01, (float)+6.85032373e-01, (float)+2.12192785e-01, },
    {(float)+3.52177943e-01, (float)+9.99168449e-01, (float)+5.89365418e-01, (float)+6.84651517e-01, (float)+2.11317389e-01, },
    {(float)+3.51482854e-01, (float)+9.99192207e-01, (float)+5.88589171e-01, (float)+6.84397613e-01, (float)+2.10458510e-01, },
    {(float)+3.50787766e-01, (float)+9.99192207e-01, (float)+5.87812925e-01, (float)+6.84143710e-01, (float)+2.09599630e-01, },
    {(float)+3.50092678e-01, (float)+9.99192207e-01, (float)+5.87036678e-01, (float)+6.83762854e-01, (float)+2.08740751e-01, },
    {(float)+3.49397590e-01, (float)+9.99215966e-01, (float)+5.86260431e-01, (float)+6.83508950e-01, (float)+2.07881871e-01, },
    {(float)+3.48702502e-01, (float)+9.99215966e-01, (float)+5.85290122e-01, (float)+6.83128094e-01, (float)+2.07022992e-01, },
    {(float)+3.48007414e-01, (float)+9.99215966e-01, (float)+5.84513875e-01, (float)+6.82874191e-01, (float)+2.06164112e-01, },
    {(float)+3.47312326e-01, (float)+9.99239724e-01, (float)+5.83737629e-01, (float)+6.82620287e-01, (float)+2.05288716e-01, },
    {(float)+3.46617238e-01, (float)+9.99239724e-01, (float)+5.82961382e-01, (float)+6.82239431e-01, (float)+2.04429836e-01, },
    {(float)+3.45922150e-01, (float)+9.99239724e-01, (float)+5.82185135e-01, (float)+6.81985527e-01, (float)+2.03570957e-01, },
    {(float)+3.45227062e-01, (float)+9.99263483e-01, (float)+5.81214826e-01, (float)+6.81731624e-01, (float)+2.02712077e-01, },
    {(float)+3.44531974e-01, (float)+9.99263483e-01, (float)+5.80438579e-01, (float)+6.81350768e-01, (float)+2.01853198e-01, },
    };
    vector<vector<float>> transpose_result = transposeMatrix(initial_data);
   // cout << "转置结果：" << transpose_result.size() << " " << transpose_result[0].size() << endl;
    vector<vector<float>> input_data= transpose_result;

    if (input_data.size() != transpose_result.size())return -1;

    vector<vector<float>> target = transpose_result;

    int function_type = 1;
    int node = 30;//第一层节点
    int laye2_neural_nodeCount = 10;
    int epochs = 10000;//训练轮次

    int target_mse = 0.000001;
    float lr = 0.05;

    // 定义第一层
    auto result_tuple_layer1 = initial_neurallayer_output(input_data, node, function_type);
    // 从 tuple 中获取返回的值
    vector<vector<float>> layer1_output = get<0>(result_tuple_layer1);
    vector<float> layer1_biasmatrix = get<1>(result_tuple_layer1);
    vector<vector<float>> layel1_weightmatrix = get<2>(result_tuple_layer1);

    vector<vector<float>> layer1_output_ini(layer1_output.size(), vector<float>(layer1_output[0].size(), 0));


    // 定义第二层
    auto result_tuple_layer2 = initial_neurallayer_output(layer1_output, laye2_neural_nodeCount, function_type);
    vector<vector<float>> layer2_output = get<0>(result_tuple_layer2);
    vector<float> layer2_biasmatrix = get<1>(result_tuple_layer2);
    vector<vector<float>> layel2_weightmatrix = get<2>(result_tuple_layer2);

    vector<vector<float>> layer2_output_ini(layer2_output.size(), vector<float>(layer2_output[0].size(), 0));

    // 输出层
    auto result_tuple_op = initial_neurallayer_output(layer2_output, target.size(), function_type);
    vector<vector<float>> output = get<0>(result_tuple_op);
    vector<float> op_biasmatrix = get<1>(result_tuple_op);
    vector<vector<float>> op_weightmatrix = get<2>(result_tuple_op);
    vector<vector<float>> error = calculateerror(output, target);

    float mse = calculateMSE(error);
    vector<vector<float>> layer2_error = calculateerror(layer2_output, layer2_output_ini);
    vector<vector<float>> layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出

    //神经网络训练主循环
    for (int epoch = 0; epoch < epochs;epoch++) {
        layer1_output = training_forward_neurallayer_output(input_data, layel1_weightmatrix, layer1_biasmatrix, node, function_type);

        // 第二层
        layer2_output = training_forward_neurallayer_output(layer1_output, layel2_weightmatrix, layer2_biasmatrix, laye2_neural_nodeCount, function_type);

        // 输出层
        output = training_forward_neurallayer_output(layer2_output, op_weightmatrix, op_biasmatrix, target.size(), function_type);

        error = calculateerror(output, target);
        mse = calculateMSE(error);
        layer2_error = calculateerror(layer2_output, layer2_output_ini);
        layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出
        update_weights(op_weightmatrix, output, error, layer2_output, lr);//参数顺序：该层权重，该层输出，该层误差，该层输入，学习率
        update_weights(layel2_weightmatrix, layer2_output, layer2_error, layer1_output, lr);
        update_weights(layel1_weightmatrix, layer1_output, layer1_error, input_data, lr);
        cout << "训练步数：" << epoch<<"  " << "MSE:" << mse << endl;
        //lr = lr * exp(-0.05*epoch);效果不佳
        if (mse < target_mse) break;

    }
    //cout << "最后一次迭代输出结果：" << endl;
    //print2DArray(output);

    return 0;
}
*/


/*
int main()
{

    const vector<vector<float>> initial_data = {
    {(float)+6.92191844e-01, (float)+9.95272036e-01, (float)+9.86803804e-01, (float)+8.32296560e-01, (float)+6.31590248e-01, },
    {(float)+6.91496756e-01, (float)+9.95272036e-01, (float)+9.86027557e-01, (float)+8.32042656e-01, (float)+6.30731369e-01, },
    {(float)+6.90801668e-01, (float)+9.95272036e-01, (float)+9.85251310e-01, (float)+8.31661800e-01, (float)+6.29872489e-01, },
    {(float)+6.90106580e-01, (float)+9.95295795e-01, (float)+9.84475063e-01, (float)+8.31407896e-01, (float)+6.28997093e-01, },
    {(float)+6.89411492e-01, (float)+9.95295795e-01, (float)+9.83698816e-01, (float)+8.31027041e-01, (float)+6.28138214e-01, },
    {(float)+6.88716404e-01, (float)+9.95295795e-01, (float)+9.82728508e-01, (float)+8.30773137e-01, (float)+6.27279334e-01, },
    {(float)+6.88021316e-01, (float)+9.95319553e-01, (float)+9.81952261e-01, (float)+8.30519233e-01, (float)+6.26420455e-01, },
    {(float)+6.87326228e-01, (float)+9.95319553e-01, (float)+9.81176014e-01, (float)+8.30138378e-01, (float)+6.25561575e-01, },
    {(float)+6.86631140e-01, (float)+9.95319553e-01, (float)+9.80399767e-01, (float)+8.29884474e-01, (float)+6.24702696e-01, },
    {(float)+6.85936052e-01, (float)+9.95343312e-01, (float)+9.79623520e-01, (float)+8.29630570e-01, (float)+6.23843816e-01, },
    {(float)+6.85240964e-01, (float)+9.95343312e-01, (float)+9.78653212e-01, (float)+8.29249714e-01, (float)+6.22968420e-01, },
    {(float)+6.84545876e-01, (float)+9.95343312e-01, (float)+9.77876965e-01, (float)+8.28995811e-01, (float)+6.22109540e-01, },
    {(float)+6.83850788e-01, (float)+9.95367071e-01, (float)+9.77100718e-01, (float)+8.28614955e-01, (float)+6.21250661e-01, },
    {(float)+6.83155700e-01, (float)+9.95367071e-01, (float)+9.76324471e-01, (float)+8.28361051e-01, (float)+6.20391781e-01, },
    {(float)+6.82460612e-01, (float)+9.95367071e-01, (float)+9.75548224e-01, (float)+8.28107147e-01, (float)+6.19532902e-01, },
    {(float)+6.81765524e-01, (float)+9.95390829e-01, (float)+9.74577916e-01, (float)+8.27726292e-01, (float)+6.18674022e-01, },
    {(float)+6.81070436e-01, (float)+9.95390829e-01, (float)+9.73801669e-01, (float)+8.27472388e-01, (float)+6.17815143e-01, },
    {(float)+6.80375348e-01, (float)+9.95390829e-01, (float)+9.73025422e-01, (float)+8.27091532e-01, (float)+6.16939746e-01, },
    {(float)+6.79680259e-01, (float)+9.95414588e-01, (float)+9.72249175e-01, (float)+8.26837629e-01, (float)+6.16080867e-01, },
    {(float)+6.78985171e-01, (float)+9.95414588e-01, (float)+9.71472928e-01, (float)+8.26583725e-01, (float)+6.15221987e-01, },
    {(float)+6.78290083e-01, (float)+9.95414588e-01, (float)+9.70502620e-01, (float)+8.26202869e-01, (float)+6.14363108e-01, },
    {(float)+6.77594995e-01, (float)+9.95438346e-01, (float)+9.69726373e-01, (float)+8.25948965e-01, (float)+6.13504228e-01, },
    {(float)+6.76899907e-01, (float)+9.95438346e-01, (float)+9.68950126e-01, (float)+8.25695062e-01, (float)+6.12645349e-01, },
    {(float)+6.76204819e-01, (float)+9.95438346e-01, (float)+9.68173879e-01, (float)+8.25314206e-01, (float)+6.11786469e-01, },
    {(float)+6.75509731e-01, (float)+9.95462105e-01, (float)+9.67397632e-01, (float)+8.25060302e-01, (float)+6.10927590e-01, },
    {(float)+6.74814643e-01, (float)+9.95462105e-01, (float)+9.66427324e-01, (float)+8.24679446e-01, (float)+6.10052193e-01, },
    {(float)+6.74119555e-01, (float)+9.95462105e-01, (float)+9.65651077e-01, (float)+8.24425543e-01, (float)+6.09193314e-01, },
    {(float)+6.73424467e-01, (float)+9.95485864e-01, (float)+9.64874830e-01, (float)+8.24171639e-01, (float)+6.08334434e-01, },
    {(float)+6.72729379e-01, (float)+9.95485864e-01, (float)+9.64098583e-01, (float)+8.23790783e-01, (float)+6.07475555e-01, },
    {(float)+6.72034291e-01, (float)+9.95485864e-01, (float)+9.63322337e-01, (float)+8.23536880e-01, (float)+6.06616675e-01, },
    {(float)+6.71223355e-01, (float)+9.95509622e-01, (float)+9.62546090e-01, (float)+8.23156024e-01, (float)+6.05757796e-01, },
    {(float)+6.70528267e-01, (float)+9.95509622e-01, (float)+9.61575781e-01, (float)+8.22902120e-01, (float)+6.04898916e-01, },
    {(float)+6.69833179e-01, (float)+9.95509622e-01, (float)+9.60799534e-01, (float)+8.22648216e-01, (float)+6.04023520e-01, },
    {(float)+6.69138091e-01, (float)+9.95533381e-01, (float)+9.60023287e-01, (float)+8.22267361e-01, (float)+6.03164641e-01, },
    {(float)+6.68443003e-01, (float)+9.95533381e-01, (float)+9.59247041e-01, (float)+8.22013457e-01, (float)+6.02305761e-01, },
    {(float)+6.67747915e-01, (float)+9.95557139e-01, (float)+9.58470794e-01, (float)+8.21759553e-01, (float)+6.01446882e-01, },
    {(float)+6.67052827e-01, (float)+9.95557139e-01, (float)+9.57500485e-01, (float)+8.21378697e-01, (float)+6.00588002e-01, },
    {(float)+6.66357739e-01, (float)+9.95557139e-01, (float)+9.56724238e-01, (float)+8.21124794e-01, (float)+5.99729123e-01, },
    {(float)+6.65662651e-01, (float)+9.95580898e-01, (float)+9.55947991e-01, (float)+8.20743938e-01, (float)+5.98870243e-01, },
    {(float)+6.64967563e-01, (float)+9.95580898e-01, (float)+9.55171745e-01, (float)+8.20490034e-01, (float)+5.97994847e-01, },
    {(float)+6.64272475e-01, (float)+9.95580898e-01, (float)+9.54395498e-01, (float)+8.20236131e-01, (float)+5.97135967e-01, },
    {(float)+6.63577386e-01, (float)+9.95604657e-01, (float)+9.53425189e-01, (float)+8.19855275e-01, (float)+5.96277088e-01, },
    {(float)+6.62882298e-01, (float)+9.95604657e-01, (float)+9.52648942e-01, (float)+8.19601371e-01, (float)+5.95418208e-01, },
    {(float)+6.62187210e-01, (float)+9.95604657e-01, (float)+9.51872696e-01, (float)+8.19347467e-01, (float)+5.94559329e-01, },
    {(float)+6.61492122e-01, (float)+9.95628415e-01, (float)+9.51096449e-01, (float)+8.18966612e-01, (float)+5.93700449e-01, },
    {(float)+6.60797034e-01, (float)+9.95628415e-01, (float)+9.50320202e-01, (float)+8.18712708e-01, (float)+5.92841570e-01, },
    {(float)+6.60101946e-01, (float)+9.95628415e-01, (float)+9.49349893e-01, (float)+8.18331852e-01, (float)+5.91966173e-01, },
    {(float)+6.59406858e-01, (float)+9.95652174e-01, (float)+9.48573646e-01, (float)+8.18077948e-01, (float)+5.91107294e-01, },
    {(float)+6.58711770e-01, (float)+9.95652174e-01, (float)+9.47797400e-01, (float)+8.17824045e-01, (float)+5.90248414e-01, },
    {(float)+6.58016682e-01, (float)+9.95652174e-01, (float)+9.47021153e-01, (float)+8.17443189e-01, (float)+5.89389535e-01, },
    {(float)+6.57321594e-01, (float)+9.95675933e-01, (float)+9.46244906e-01, (float)+8.17189285e-01, (float)+5.88530655e-01, },
    {(float)+6.56626506e-01, (float)+9.95675933e-01, (float)+9.45274597e-01, (float)+8.16808430e-01, (float)+5.87671776e-01, },
    {(float)+6.55931418e-01, (float)+9.95675933e-01, (float)+9.44498350e-01, (float)+8.16554526e-01, (float)+5.86812896e-01, },
    {(float)+6.55236330e-01, (float)+9.95699691e-01, (float)+9.43722104e-01, (float)+8.16300622e-01, (float)+5.85937500e-01, },
    {(float)+6.54541242e-01, (float)+9.95699691e-01, (float)+9.42945857e-01, (float)+8.15919766e-01, (float)+5.85078621e-01, },
    {(float)+6.53846154e-01, (float)+9.95699691e-01, (float)+9.42169610e-01, (float)+8.15665863e-01, (float)+5.84219741e-01, },
    {(float)+6.53151066e-01, (float)+9.95723450e-01, (float)+9.41199301e-01, (float)+8.15411959e-01, (float)+5.83360862e-01, },
    {(float)+6.52455978e-01, (float)+9.95723450e-01, (float)+9.40423055e-01, (float)+8.15031103e-01, (float)+5.82501982e-01, },
    {(float)+6.51760890e-01, (float)+9.95723450e-01, (float)+9.39646808e-01, (float)+8.14777199e-01, (float)+5.81643103e-01, },
    {(float)+6.51065802e-01, (float)+9.95747208e-01, (float)+9.38870561e-01, (float)+8.14396344e-01, (float)+5.80784223e-01, },
    {(float)+6.50370714e-01, (float)+9.95747208e-01, (float)+9.38094314e-01, (float)+8.14142440e-01, (float)+5.79908827e-01, },
    {(float)+6.49675626e-01, (float)+9.95747208e-01, (float)+9.37124005e-01, (float)+8.13888536e-01, (float)+5.79049947e-01, },
    {(float)+6.48980538e-01, (float)+9.95770967e-01, (float)+9.36347759e-01, (float)+8.13507681e-01, (float)+5.78191068e-01, },
    {(float)+6.48285449e-01, (float)+9.95770967e-01, (float)+9.35571512e-01, (float)+8.13253777e-01, (float)+5.77332188e-01, },
    {(float)+6.47590361e-01, (float)+9.95770967e-01, (float)+9.34795265e-01, (float)+8.12872921e-01, (float)+5.76473309e-01, },
    {(float)+6.46895273e-01, (float)+9.95794726e-01, (float)+9.34019018e-01, (float)+8.12619017e-01, (float)+5.75614429e-01, },
    {(float)+6.46200185e-01, (float)+9.95794726e-01, (float)+9.33048709e-01, (float)+8.12365114e-01, (float)+5.74755550e-01, },
    {(float)+6.45505097e-01, (float)+9.95794726e-01, (float)+9.32272463e-01, (float)+8.11984258e-01, (float)+5.73880153e-01, },
    {(float)+6.44810009e-01, (float)+9.95818484e-01, (float)+9.31496216e-01, (float)+8.11730354e-01, (float)+5.73021274e-01, },
    {(float)+6.44114921e-01, (float)+9.95818484e-01, (float)+9.30719969e-01, (float)+8.11476450e-01, (float)+5.72162394e-01, },
    {(float)+6.43419833e-01, (float)+9.95818484e-01, (float)+9.29943722e-01, (float)+8.11095595e-01, (float)+5.71303515e-01, },
    {(float)+6.42724745e-01, (float)+9.95842243e-01, (float)+9.28973414e-01, (float)+8.10841691e-01, (float)+5.70444635e-01, },
    {(float)+6.42029657e-01, (float)+9.95842243e-01, (float)+9.28197167e-01, (float)+8.10460835e-01, (float)+5.69585756e-01, },
    {(float)+6.41334569e-01, (float)+9.95842243e-01, (float)+9.27420920e-01, (float)+8.10206932e-01, (float)+5.68726876e-01, },
    {(float)+6.40639481e-01, (float)+9.95866001e-01, (float)+9.26644673e-01, (float)+8.09953028e-01, (float)+5.67867997e-01, },
    {(float)+6.39944393e-01, (float)+9.95866001e-01, (float)+9.25868426e-01, (float)+8.09572172e-01, (float)+5.66992600e-01, },
    {(float)+6.39249305e-01, (float)+9.95866001e-01, (float)+9.24898118e-01, (float)+8.09318268e-01, (float)+5.66133721e-01, },
    {(float)+6.38554217e-01, (float)+9.95889760e-01, (float)+9.24121871e-01, (float)+8.09064365e-01, (float)+5.65274841e-01, },
    {(float)+6.37859129e-01, (float)+9.95889760e-01, (float)+9.23345624e-01, (float)+8.08683509e-01, (float)+5.64415962e-01, },
    {(float)+6.37164041e-01, (float)+9.95889760e-01, (float)+9.22569377e-01, (float)+8.08429605e-01, (float)+5.63557082e-01, },
    {(float)+6.36468953e-01, (float)+9.95913519e-01, (float)+9.21793130e-01, (float)+8.08048750e-01, (float)+5.62698203e-01, },
    {(float)+6.35773865e-01, (float)+9.95913519e-01, (float)+9.20822822e-01, (float)+8.07794846e-01, (float)+5.61839323e-01, },
    {(float)+6.35078777e-01, (float)+9.95913519e-01, (float)+9.20046575e-01, (float)+8.07540942e-01, (float)+5.60963927e-01, },
    {(float)+6.34383689e-01, (float)+9.95937277e-01, (float)+9.19270328e-01, (float)+8.07160086e-01, (float)+5.60105048e-01, },
    {(float)+6.33688601e-01, (float)+9.95937277e-01, (float)+9.18494081e-01, (float)+8.06906183e-01, (float)+5.59246168e-01, },
    {(float)+6.32993513e-01, (float)+9.95937277e-01, (float)+9.17717834e-01, (float)+8.06525327e-01, (float)+5.58387289e-01, },
    {(float)+6.32298424e-01, (float)+9.95961036e-01, (float)+9.16747526e-01, (float)+8.06271423e-01, (float)+5.57528409e-01, },
    {(float)+6.31603336e-01, (float)+9.95961036e-01, (float)+9.15971279e-01, (float)+8.06017519e-01, (float)+5.56669530e-01, },
    {(float)+6.30908248e-01, (float)+9.95961036e-01, (float)+9.15195032e-01, (float)+8.05636664e-01, (float)+5.55810650e-01, },
    {(float)+6.30213160e-01, (float)+9.95984794e-01, (float)+9.14418785e-01, (float)+8.05382760e-01, (float)+5.54935254e-01, },
    {(float)+6.29518072e-01, (float)+9.95984794e-01, (float)+9.13642538e-01, (float)+8.05128856e-01, (float)+5.54076374e-01, },
    {(float)+6.28822984e-01, (float)+9.95984794e-01, (float)+9.12672230e-01, (float)+8.04748001e-01, (float)+5.53217495e-01, },
    {(float)+6.28127896e-01, (float)+9.96008553e-01, (float)+9.11895983e-01, (float)+8.04494097e-01, (float)+5.52358615e-01, },
    {(float)+6.27432808e-01, (float)+9.96008553e-01, (float)+9.11119736e-01, (float)+8.04113241e-01, (float)+5.51499736e-01, },
    {(float)+6.26737720e-01, (float)+9.96008553e-01, (float)+9.10343489e-01, (float)+8.03859337e-01, (float)+5.50640856e-01, },
    {(float)+6.26042632e-01, (float)+9.96032312e-01, (float)+9.09567242e-01, (float)+8.03605434e-01, (float)+5.49781977e-01, },
    {(float)+6.25347544e-01, (float)+9.96032312e-01, (float)+9.08790996e-01, (float)+8.03224578e-01, (float)+5.48906580e-01, },
    {(float)+6.24652456e-01, (float)+9.96032312e-01, (float)+9.07820687e-01, (float)+8.02970674e-01, (float)+5.48047701e-01, },
    {(float)+6.23957368e-01, (float)+9.96056070e-01, (float)+9.07044440e-01, (float)+8.02589818e-01, (float)+5.47188821e-01, },
    {(float)+6.23262280e-01, (float)+9.96056070e-01, (float)+9.06268193e-01, (float)+8.02335915e-01, (float)+5.46329942e-01, },
    {(float)+6.22567192e-01, (float)+9.96056070e-01, (float)+9.05491946e-01, (float)+8.02082011e-01, (float)+5.45471062e-01, },
    {(float)+6.21872104e-01, (float)+9.96079829e-01, (float)+9.04715700e-01, (float)+8.01701155e-01, (float)+5.44612183e-01, },
    {(float)+6.21177016e-01, (float)+9.96079829e-01, (float)+9.03745391e-01, (float)+8.01447251e-01, (float)+5.43753303e-01, },
    {(float)+6.20481928e-01, (float)+9.96079829e-01, (float)+9.02969144e-01, (float)+8.01193348e-01, (float)+5.42877907e-01, },
    {(float)+6.19786840e-01, (float)+9.96103588e-01, (float)+9.02192897e-01, (float)+8.00812492e-01, (float)+5.42019027e-01, },
    {(float)+6.18975904e-01, (float)+9.96103588e-01, (float)+9.01416650e-01, (float)+8.00558588e-01, (float)+5.41160148e-01, },
    {(float)+6.18280816e-01, (float)+9.96103588e-01, (float)+9.00640404e-01, (float)+8.00177733e-01, (float)+5.40301268e-01, },
    {(float)+6.17585728e-01, (float)+9.96127346e-01, (float)+8.99670095e-01, (float)+7.99923829e-01, (float)+5.39442389e-01, },
    {(float)+6.16890639e-01, (float)+9.96127346e-01, (float)+8.98893848e-01, (float)+7.99669925e-01, (float)+5.38583510e-01, },
    {(float)+6.16195551e-01, (float)+9.96127346e-01, (float)+8.98117601e-01, (float)+7.99289069e-01, (float)+5.37724630e-01, },
    {(float)+6.15500463e-01, (float)+9.96151105e-01, (float)+8.97341355e-01, (float)+7.99035166e-01, (float)+5.36849234e-01, },
    {(float)+6.14805375e-01, (float)+9.96151105e-01, (float)+8.96565108e-01, (float)+7.98781262e-01, (float)+5.35990354e-01, },
    {(float)+6.14110287e-01, (float)+9.96151105e-01, (float)+8.95594799e-01, (float)+7.98400406e-01, (float)+5.35131475e-01, },
    {(float)+6.13415199e-01, (float)+9.96174863e-01, (float)+8.94818552e-01, (float)+7.98146502e-01, (float)+5.34272595e-01, },
    {(float)+6.12720111e-01, (float)+9.96174863e-01, (float)+8.94042305e-01, (float)+7.97765647e-01, (float)+5.33413716e-01, },
    {(float)+6.12025023e-01, (float)+9.96174863e-01, (float)+8.93266059e-01, (float)+7.97511743e-01, (float)+5.32554836e-01, },
    {(float)+6.11329935e-01, (float)+9.96198622e-01, (float)+8.92489812e-01, (float)+7.97257839e-01, (float)+5.31695957e-01, },
    {(float)+6.10634847e-01, (float)+9.96198622e-01, (float)+8.91519503e-01, (float)+7.96876984e-01, (float)+5.30820560e-01, },
    {(float)+6.09939759e-01, (float)+9.96198622e-01, (float)+8.90743256e-01, (float)+7.96623080e-01, (float)+5.29961681e-01, },
    {(float)+6.09244671e-01, (float)+9.96222381e-01, (float)+8.89967010e-01, (float)+7.96242224e-01, (float)+5.29102801e-01, },
    {(float)+6.08549583e-01, (float)+9.96222381e-01, (float)+8.89190763e-01, (float)+7.95988320e-01, (float)+5.28243922e-01, },
    {(float)+6.07854495e-01, (float)+9.96246139e-01, (float)+8.88414516e-01, (float)+7.95734417e-01, (float)+5.27385042e-01, },
    {(float)+6.07159407e-01, (float)+9.96246139e-01, (float)+8.87444207e-01, (float)+7.95353561e-01, (float)+5.26526163e-01, },
    {(float)+6.06464319e-01, (float)+9.96246139e-01, (float)+8.86667960e-01, (float)+7.95099657e-01, (float)+5.25667283e-01, },
    {(float)+6.05769231e-01, (float)+9.96269898e-01, (float)+8.85891714e-01, (float)+7.94845753e-01, (float)+5.24791887e-01, },
    {(float)+6.05074143e-01, (float)+9.96269898e-01, (float)+8.85115467e-01, (float)+7.94464898e-01, (float)+5.23933007e-01, },
    {(float)+6.04379055e-01, (float)+9.96269898e-01, (float)+8.84339220e-01, (float)+7.94210994e-01, (float)+5.23074128e-01, },
    {(float)+6.03683967e-01, (float)+9.96293656e-01, (float)+8.83368911e-01, (float)+7.93830138e-01, (float)+5.22215248e-01, },
    {(float)+6.02988879e-01, (float)+9.96293656e-01, (float)+8.82592664e-01, (float)+7.93576235e-01, (float)+5.21356369e-01, },
    {(float)+6.02293791e-01, (float)+9.96293656e-01, (float)+8.81816418e-01, (float)+7.93322331e-01, (float)+5.20497489e-01, },
    {(float)+6.01598703e-01, (float)+9.96317415e-01, (float)+8.81040171e-01, (float)+7.92941475e-01, (float)+5.19638610e-01, },
    {(float)+6.00903614e-01, (float)+9.96317415e-01, (float)+8.80263924e-01, (float)+7.92687571e-01, (float)+5.18779730e-01, },
    {(float)+6.00208526e-01, (float)+9.96317415e-01, (float)+8.79293615e-01, (float)+7.92306716e-01, (float)+5.17904334e-01, },
    {(float)+5.99513438e-01, (float)+9.96341174e-01, (float)+8.78517369e-01, (float)+7.92052812e-01, (float)+5.17045455e-01, },
    {(float)+5.98818350e-01, (float)+9.96341174e-01, (float)+8.77741122e-01, (float)+7.91798908e-01, (float)+5.16186575e-01, },
    {(float)+5.98123262e-01, (float)+9.96341174e-01, (float)+8.76964875e-01, (float)+7.91418053e-01, (float)+5.15327696e-01, },
    {(float)+5.97428174e-01, (float)+9.96364932e-01, (float)+8.76188628e-01, (float)+7.91164149e-01, (float)+5.14468816e-01, },
    {(float)+5.96733086e-01, (float)+9.96364932e-01, (float)+8.75218319e-01, (float)+7.90910245e-01, (float)+5.13609937e-01, },
    {(float)+5.96037998e-01, (float)+9.96364932e-01, (float)+8.74442073e-01, (float)+7.90529389e-01, (float)+5.12751057e-01, },
    {(float)+5.95342910e-01, (float)+9.96388691e-01, (float)+8.73665826e-01, (float)+7.90275486e-01, (float)+5.11875661e-01, },
    {(float)+5.94647822e-01, (float)+9.96388691e-01, (float)+8.72889579e-01, (float)+7.89894630e-01, (float)+5.11016781e-01, },
    {(float)+5.93952734e-01, (float)+9.96388691e-01, (float)+8.72113332e-01, (float)+7.89640726e-01, (float)+5.10157902e-01, },
    {(float)+5.93257646e-01, (float)+9.96412450e-01, (float)+8.71143023e-01, (float)+7.89386822e-01, (float)+5.09299022e-01, },
    {(float)+5.92562558e-01, (float)+9.96412450e-01, (float)+8.70366777e-01, (float)+7.89005967e-01, (float)+5.08440143e-01, },
    {(float)+5.91867470e-01, (float)+9.96412450e-01, (float)+8.69590530e-01, (float)+7.88752063e-01, (float)+5.07581263e-01, },
    {(float)+5.91172382e-01, (float)+9.96436208e-01, (float)+8.68814283e-01, (float)+7.88498159e-01, (float)+5.06722384e-01, },
    {(float)+5.90477294e-01, (float)+9.96436208e-01, (float)+8.68038036e-01, (float)+7.88117304e-01, (float)+5.05846987e-01, },
    {(float)+5.89782206e-01, (float)+9.96436208e-01, (float)+8.67067728e-01, (float)+7.87863400e-01, (float)+5.04988108e-01, },
    {(float)+5.89087118e-01, (float)+9.96459967e-01, (float)+8.66291481e-01, (float)+7.87482544e-01, (float)+5.04129228e-01, },
    {(float)+5.88392030e-01, (float)+9.96459967e-01, (float)+8.65515234e-01, (float)+7.87228640e-01, (float)+5.03270349e-01, },
    {(float)+5.87696942e-01, (float)+9.96459967e-01, (float)+8.64738987e-01, (float)+7.86974737e-01, (float)+5.02411469e-01, },
    {(float)+5.87001854e-01, (float)+9.96483725e-01, (float)+8.63962740e-01, (float)+7.86593881e-01, (float)+5.01552590e-01, },
    {(float)+5.86306766e-01, (float)+9.96483725e-01, (float)+8.62992432e-01, (float)+7.86339977e-01, (float)+5.00693710e-01, },
    {(float)+5.85611677e-01, (float)+9.96483725e-01, (float)+8.62216185e-01, (float)+7.85959121e-01, (float)+4.99818314e-01, },
    {(float)+5.84916589e-01, (float)+9.96507484e-01, (float)+8.61439938e-01, (float)+7.85705218e-01, (float)+4.98959434e-01, },
    {(float)+5.84221501e-01, (float)+9.96507484e-01, (float)+8.60663691e-01, (float)+7.85451314e-01, (float)+4.98100555e-01, },
    {(float)+5.83526413e-01, (float)+9.96507484e-01, (float)+8.59887444e-01, (float)+7.85070458e-01, (float)+4.97241675e-01, },
    {(float)+5.82831325e-01, (float)+9.96531243e-01, (float)+8.58917136e-01, (float)+7.84816555e-01, (float)+4.96382796e-01, },
    {(float)+5.82136237e-01, (float)+9.96531243e-01, (float)+8.58140889e-01, (float)+7.84562651e-01, (float)+4.95523916e-01, },
    {(float)+5.81441149e-01, (float)+9.96531243e-01, (float)+8.57364642e-01, (float)+7.84181795e-01, (float)+4.94665037e-01, },
    {(float)+5.80746061e-01, (float)+9.96555001e-01, (float)+8.56588395e-01, (float)+7.83927891e-01, (float)+4.93789641e-01, },
    {(float)+5.80050973e-01, (float)+9.96555001e-01, (float)+8.55812148e-01, (float)+7.83547036e-01, (float)+4.92930761e-01, },
    {(float)+5.79355885e-01, (float)+9.96555001e-01, (float)+8.55035901e-01, (float)+7.83293132e-01, (float)+4.92071882e-01, },
    {(float)+5.78660797e-01, (float)+9.96578760e-01, (float)+8.54065593e-01, (float)+7.83039228e-01, (float)+4.91213002e-01, },
    {(float)+5.77965709e-01, (float)+9.96578760e-01, (float)+8.53289346e-01, (float)+7.82658372e-01, (float)+4.90354123e-01, },
    {(float)+5.77270621e-01, (float)+9.96578760e-01, (float)+8.52513099e-01, (float)+7.82404469e-01, (float)+4.89495243e-01, },
    {(float)+5.76575533e-01, (float)+9.96602518e-01, (float)+8.51736852e-01, (float)+7.82023613e-01, (float)+4.88636364e-01, },
    {(float)+5.75880445e-01, (float)+9.96602518e-01, (float)+8.50960605e-01, (float)+7.81769709e-01, (float)+4.87760967e-01, },
    {(float)+5.75185357e-01, (float)+9.96602518e-01, (float)+8.49990297e-01, (float)+7.81515806e-01, (float)+4.86902088e-01, },
    {(float)+5.74490269e-01, (float)+9.96626277e-01, (float)+8.49214050e-01, (float)+7.81134950e-01, (float)+4.86043208e-01, },
    {(float)+5.73795181e-01, (float)+9.96626277e-01, (float)+8.48437803e-01, (float)+7.80881046e-01, (float)+4.85184329e-01, },
    {(float)+5.73100093e-01, (float)+9.96626277e-01, (float)+8.47661556e-01, (float)+7.80627142e-01, (float)+4.84325449e-01, },
    {(float)+5.72405005e-01, (float)+9.96650036e-01, (float)+8.46885310e-01, (float)+7.80246287e-01, (float)+4.83466570e-01, },
    {(float)+5.71709917e-01, (float)+9.96650036e-01, (float)+8.45915001e-01, (float)+7.79992383e-01, (float)+4.82607690e-01, },
    {(float)+5.71014829e-01, (float)+9.96650036e-01, (float)+8.45138754e-01, (float)+7.79611527e-01, (float)+4.81732294e-01, },
    {(float)+5.70319741e-01, (float)+9.96673794e-01, (float)+8.44362507e-01, (float)+7.79357623e-01, (float)+4.80873414e-01, },
    {(float)+5.69624652e-01, (float)+9.96673794e-01, (float)+8.43586260e-01, (float)+7.79103720e-01, (float)+4.80014535e-01, },
    {(float)+5.68929564e-01, (float)+9.96673794e-01, (float)+8.42810014e-01, (float)+7.78722864e-01, (float)+4.79155655e-01, },
    {(float)+5.68234476e-01, (float)+9.96697553e-01, (float)+8.41839705e-01, (float)+7.78468960e-01, (float)+4.78296776e-01, },
    {(float)+5.67539388e-01, (float)+9.96697553e-01, (float)+8.41063458e-01, (float)+7.78215056e-01, (float)+4.77437896e-01, },
    {(float)+5.66728452e-01, (float)+9.96697553e-01, (float)+8.40287211e-01, (float)+7.77834201e-01, (float)+4.76579017e-01, },
    {(float)+5.66033364e-01, (float)+9.96721311e-01, (float)+8.39510964e-01, (float)+7.77580297e-01, (float)+4.75703621e-01, },
    {(float)+5.65338276e-01, (float)+9.96721311e-01, (float)+8.38734718e-01, (float)+7.77199441e-01, (float)+4.74844741e-01, },
    {(float)+5.64643188e-01, (float)+9.96721311e-01, (float)+8.37764409e-01, (float)+7.76945538e-01, (float)+4.73985862e-01, },
    {(float)+5.63948100e-01, (float)+9.96745070e-01, (float)+8.36988162e-01, (float)+7.76691634e-01, (float)+4.73126982e-01, },
    {(float)+5.63253012e-01, (float)+9.96745070e-01, (float)+8.36211915e-01, (float)+7.76310778e-01, (float)+4.72268103e-01, },
    {(float)+5.62557924e-01, (float)+9.96745070e-01, (float)+8.35435669e-01, (float)+7.76056874e-01, (float)+4.71409223e-01, },
    {(float)+5.61862836e-01, (float)+9.96768829e-01, (float)+8.34659422e-01, (float)+7.75676019e-01, (float)+4.70550344e-01, },
    {(float)+5.61167748e-01, (float)+9.96768829e-01, (float)+8.33689113e-01, (float)+7.75422115e-01, (float)+4.69691464e-01, },
    {(float)+5.60472660e-01, (float)+9.96768829e-01, (float)+8.32912866e-01, (float)+7.75168211e-01, (float)+4.68816068e-01, },
    {(float)+5.59777572e-01, (float)+9.96792587e-01, (float)+8.32136619e-01, (float)+7.74787356e-01, (float)+4.67957188e-01, },
    {(float)+5.59082484e-01, (float)+9.96792587e-01, (float)+8.31360373e-01, (float)+7.74533452e-01, (float)+4.67098309e-01, },
    {(float)+5.58387396e-01, (float)+9.96792587e-01, (float)+8.30584126e-01, (float)+7.74279548e-01, (float)+4.66239429e-01, },
    {(float)+5.57692308e-01, (float)+9.96816346e-01, (float)+8.29613817e-01, (float)+7.73898692e-01, (float)+4.65380550e-01, },
    {(float)+5.56997220e-01, (float)+9.96816346e-01, (float)+8.28837570e-01, (float)+7.73644789e-01, (float)+4.64521670e-01, },
    {(float)+5.56302132e-01, (float)+9.96816346e-01, (float)+8.28061324e-01, (float)+7.73263933e-01, (float)+4.63662791e-01, },
    {(float)+5.55607044e-01, (float)+9.96840105e-01, (float)+8.27285077e-01, (float)+7.73010029e-01, (float)+4.62787394e-01, },
    {(float)+5.54911956e-01, (float)+9.96840105e-01, (float)+8.26508830e-01, (float)+7.72756125e-01, (float)+4.61928515e-01, },
    {(float)+5.54216867e-01, (float)+9.96840105e-01, (float)+8.25538521e-01, (float)+7.72375270e-01, (float)+4.61069635e-01, },
    {(float)+5.53521779e-01, (float)+9.96863863e-01, (float)+8.24762274e-01, (float)+7.72121366e-01, (float)+4.60210756e-01, },
    {(float)+5.52826691e-01, (float)+9.96863863e-01, (float)+8.23986028e-01, (float)+7.71740510e-01, (float)+4.59351876e-01, },
    {(float)+5.52131603e-01, (float)+9.96863863e-01, (float)+8.23209781e-01, (float)+7.71486607e-01, (float)+4.58492997e-01, },
    {(float)+5.51436515e-01, (float)+9.96887622e-01, (float)+8.22433534e-01, (float)+7.71232703e-01, (float)+4.57634117e-01, },
    {(float)+5.50741427e-01, (float)+9.96887622e-01, (float)+8.21463225e-01, (float)+7.70851847e-01, (float)+4.56758721e-01, },
    {(float)+5.50046339e-01, (float)+9.96887622e-01, (float)+8.20686978e-01, (float)+7.70597943e-01, (float)+4.55899841e-01, },
    {(float)+5.49351251e-01, (float)+9.96911380e-01, (float)+8.19910732e-01, (float)+7.70344040e-01, (float)+4.55040962e-01, },
    {(float)+5.48656163e-01, (float)+9.96911380e-01, (float)+8.19134485e-01, (float)+7.69963184e-01, (float)+4.54182082e-01, },
    {(float)+5.47961075e-01, (float)+9.96935139e-01, (float)+8.18358238e-01, (float)+7.69709280e-01, (float)+4.53323203e-01, },
    {(float)+5.47265987e-01, (float)+9.96935139e-01, (float)+8.17387929e-01, (float)+7.69328425e-01, (float)+4.52464323e-01, },
    {(float)+5.46570899e-01, (float)+9.96935139e-01, (float)+8.16611683e-01, (float)+7.69074521e-01, (float)+4.51605444e-01, },
    {(float)+5.45875811e-01, (float)+9.96958898e-01, (float)+8.15835436e-01, (float)+7.68820617e-01, (float)+4.50730048e-01, },
    {(float)+5.45180723e-01, (float)+9.96958898e-01, (float)+8.15059189e-01, (float)+7.68439761e-01, (float)+4.49871168e-01, },
    {(float)+5.44485635e-01, (float)+9.96958898e-01, (float)+8.14282942e-01, (float)+7.68185858e-01, (float)+4.49012289e-01, },
    {(float)+5.43790547e-01, (float)+9.96982656e-01, (float)+8.13312633e-01, (float)+7.67931954e-01, (float)+4.48153409e-01, },
    {(float)+5.43095459e-01, (float)+9.96982656e-01, (float)+8.12536387e-01, (float)+7.67551098e-01, (float)+4.47294530e-01, },
    {(float)+5.42400371e-01, (float)+9.96982656e-01, (float)+8.11760140e-01, (float)+7.67297194e-01, (float)+4.46435650e-01, },
    {(float)+5.41705283e-01, (float)+9.97006415e-01, (float)+8.10983893e-01, (float)+7.66916339e-01, (float)+4.45576771e-01, },
    {(float)+5.41010195e-01, (float)+9.97006415e-01, (float)+8.10207646e-01, (float)+7.66662435e-01, (float)+4.44701374e-01, },
    {(float)+5.40315107e-01, (float)+9.97006415e-01, (float)+8.09237337e-01, (float)+7.66408531e-01, (float)+4.43842495e-01, },
    {(float)+5.39620019e-01, (float)+9.97030173e-01, (float)+8.08461091e-01, (float)+7.66027676e-01, (float)+4.42983615e-01, },
    {(float)+5.38924930e-01, (float)+9.97030173e-01, (float)+8.07684844e-01, (float)+7.65773772e-01, (float)+4.42124736e-01, },
    {(float)+5.38229842e-01, (float)+9.97030173e-01, (float)+8.06908597e-01, (float)+7.65392916e-01, (float)+4.41265856e-01, },
    {(float)+5.37534754e-01, (float)+9.97053932e-01, (float)+8.06132350e-01, (float)+7.65139012e-01, (float)+4.40406977e-01, },
    {(float)+5.36839666e-01, (float)+9.97053932e-01, (float)+8.05162042e-01, (float)+7.64885109e-01, (float)+4.39548097e-01, },
    {(float)+5.36144578e-01, (float)+9.97053932e-01, (float)+8.04385795e-01, (float)+7.64504253e-01, (float)+4.38672701e-01, },
    {(float)+5.35449490e-01, (float)+9.97077691e-01, (float)+8.03609548e-01, (float)+7.64250349e-01, (float)+4.37813821e-01, },
    {(float)+5.34754402e-01, (float)+9.97077691e-01, (float)+8.02833301e-01, (float)+7.63996445e-01, (float)+4.36954942e-01, },
    {(float)+5.34059314e-01, (float)+9.97077691e-01, (float)+8.02057054e-01, (float)+7.63615590e-01, (float)+4.36096062e-01, },
    {(float)+5.33364226e-01, (float)+9.97101449e-01, (float)+8.01280807e-01, (float)+7.63361686e-01, (float)+4.35237183e-01, },
    {(float)+5.32669138e-01, (float)+9.97101449e-01, (float)+8.00310499e-01, (float)+7.62980830e-01, (float)+4.34378303e-01, },
    {(float)+5.31974050e-01, (float)+9.97101449e-01, (float)+7.99534252e-01, (float)+7.62726926e-01, (float)+4.33519424e-01, },
    {(float)+5.31278962e-01, (float)+9.97125208e-01, (float)+7.98758005e-01, (float)+7.62473023e-01, (float)+4.32644027e-01, },
    {(float)+5.30583874e-01, (float)+9.97125208e-01, (float)+7.97981758e-01, (float)+7.62092167e-01, (float)+4.31785148e-01, },
    {(float)+5.29888786e-01, (float)+9.97125208e-01, (float)+7.97205511e-01, (float)+7.61838263e-01, (float)+4.30926268e-01, },
    {(float)+5.29193698e-01, (float)+9.97148967e-01, (float)+7.96235203e-01, (float)+7.61457408e-01, (float)+4.30067389e-01, },
    {(float)+5.28498610e-01, (float)+9.97148967e-01, (float)+7.95458956e-01, (float)+7.61203504e-01, (float)+4.29208510e-01, },
    {(float)+5.27803522e-01, (float)+9.97148967e-01, (float)+7.94682709e-01, (float)+7.60949600e-01, (float)+4.28349630e-01, },
    {(float)+5.27108434e-01, (float)+9.97172725e-01, (float)+7.93906462e-01, (float)+7.60568744e-01, (float)+4.27490751e-01, },
    {(float)+5.26413346e-01, (float)+9.97172725e-01, (float)+7.93130215e-01, (float)+7.60314841e-01, (float)+4.26631871e-01, },
    {(float)+5.25718258e-01, (float)+9.97172725e-01, (float)+7.92159907e-01, (float)+7.60060937e-01, (float)+4.25756475e-01, },
    {(float)+5.25023170e-01, (float)+9.97196484e-01, (float)+7.91383660e-01, (float)+7.59680081e-01, (float)+4.24897595e-01, },
    {(float)+5.24328082e-01, (float)+9.97196484e-01, (float)+7.90607413e-01, (float)+7.59426177e-01, (float)+4.24038716e-01, },
    {(float)+5.23632994e-01, (float)+9.97196484e-01, (float)+7.89831166e-01, (float)+7.59045322e-01, (float)+4.23179836e-01, },
    {(float)+5.22937905e-01, (float)+9.97220242e-01, (float)+7.89054919e-01, (float)+7.58791418e-01, (float)+4.22320957e-01, },
    {(float)+5.22242817e-01, (float)+9.97220242e-01, (float)+7.88084611e-01, (float)+7.58537514e-01, (float)+4.21462077e-01, },
    {(float)+5.21547729e-01, (float)+9.97220242e-01, (float)+7.87308364e-01, (float)+7.58156659e-01, (float)+4.20603198e-01, },
    {(float)+5.20852641e-01, (float)+9.97244001e-01, (float)+7.86532117e-01, (float)+7.57902755e-01, (float)+4.19727801e-01, },
    {(float)+5.20157553e-01, (float)+9.97244001e-01, (float)+7.85755870e-01, (float)+7.57648851e-01, (float)+4.18868922e-01, },
    {(float)+5.19462465e-01, (float)+9.97244001e-01, (float)+7.84979624e-01, (float)+7.57267995e-01, (float)+4.18010042e-01, },
    {(float)+5.18767377e-01, (float)+9.97267760e-01, (float)+7.84009315e-01, (float)+7.57014092e-01, (float)+4.17151163e-01, },
    {(float)+5.18072289e-01, (float)+9.97267760e-01, (float)+7.83233068e-01, (float)+7.56633236e-01, (float)+4.16292283e-01, },
    {(float)+5.17377201e-01, (float)+9.97267760e-01, (float)+7.82456821e-01, (float)+7.56379332e-01, (float)+4.15433404e-01, },
    {(float)+5.16682113e-01, (float)+9.97291518e-01, (float)+7.81680574e-01, (float)+7.56125428e-01, (float)+4.14574524e-01, },
    {(float)+5.15987025e-01, (float)+9.97291518e-01, (float)+7.80904328e-01, (float)+7.55744573e-01, (float)+4.13699128e-01, },
    {(float)+5.15291937e-01, (float)+9.97291518e-01, (float)+7.79934019e-01, (float)+7.55490669e-01, (float)+4.12840248e-01, },
    {(float)+5.14481001e-01, (float)+9.97315277e-01, (float)+7.79157772e-01, (float)+7.55109813e-01, (float)+4.11981369e-01, },
    {(float)+5.13785913e-01, (float)+9.97315277e-01, (float)+7.78381525e-01, (float)+7.54855910e-01, (float)+4.11122489e-01, },
    {(float)+5.13090825e-01, (float)+9.97315277e-01, (float)+7.77605278e-01, (float)+7.54602006e-01, (float)+4.10263610e-01, },
    {(float)+5.12395737e-01, (float)+9.97339035e-01, (float)+7.76829032e-01, (float)+7.54221150e-01, (float)+4.09404730e-01, },
    {(float)+5.11700649e-01, (float)+9.97339035e-01, (float)+7.75858723e-01, (float)+7.53967246e-01, (float)+4.08545851e-01, },
    {(float)+5.11005561e-01, (float)+9.97339035e-01, (float)+7.75082476e-01, (float)+7.53713343e-01, (float)+4.07670455e-01, },
    {(float)+5.10310473e-01, (float)+9.97362794e-01, (float)+7.74306229e-01, (float)+7.53332487e-01, (float)+4.06811575e-01, },
    {(float)+5.09615385e-01, (float)+9.97362794e-01, (float)+7.73529983e-01, (float)+7.53078583e-01, (float)+4.05952696e-01, },
    {(float)+5.08920297e-01, (float)+9.97362794e-01, (float)+7.72753736e-01, (float)+7.52697728e-01, (float)+4.05093816e-01, },
    {(float)+5.08225209e-01, (float)+9.97386553e-01, (float)+7.71783427e-01, (float)+7.52443824e-01, (float)+4.04234937e-01, },
    {(float)+5.07530120e-01, (float)+9.97386553e-01, (float)+7.71007180e-01, (float)+7.52189920e-01, (float)+4.03376057e-01, },
    {(float)+5.06835032e-01, (float)+9.97386553e-01, (float)+7.70230933e-01, (float)+7.51809064e-01, (float)+4.02517178e-01, },
    {(float)+5.06139944e-01, (float)+9.97410311e-01, (float)+7.69454687e-01, (float)+7.51555161e-01, (float)+4.01641781e-01, },
    {(float)+5.05444856e-01, (float)+9.97410311e-01, (float)+7.68678440e-01, (float)+7.51174305e-01, (float)+4.00782902e-01, },
    {(float)+5.04749768e-01, (float)+9.97410311e-01, (float)+7.67708131e-01, (float)+7.50920401e-01, (float)+3.99924022e-01, },
    {(float)+5.04054680e-01, (float)+9.97434070e-01, (float)+7.66931884e-01, (float)+7.50666497e-01, (float)+3.99065143e-01, },
    {(float)+5.03359592e-01, (float)+9.97434070e-01, (float)+7.66155637e-01, (float)+7.50285642e-01, (float)+3.98206263e-01, },
    {(float)+5.02664504e-01, (float)+9.97434070e-01, (float)+7.65379391e-01, (float)+7.50031738e-01, (float)+3.97347384e-01, },
    {(float)+5.01969416e-01, (float)+9.97457828e-01, (float)+7.64603144e-01, (float)+7.49777834e-01, (float)+3.96488504e-01, },
    {(float)+5.01274328e-01, (float)+9.97457828e-01, (float)+7.63632835e-01, (float)+7.49396979e-01, (float)+3.95613108e-01, },
    {(float)+5.00579240e-01, (float)+9.97457828e-01, (float)+7.62856588e-01, (float)+7.49143075e-01, (float)+3.94754228e-01, },
    {(float)+4.99884152e-01, (float)+9.97481587e-01, (float)+7.62080342e-01, (float)+7.48762219e-01, (float)+3.93895349e-01, },
    {(float)+4.99189064e-01, (float)+9.97481587e-01, (float)+7.61304095e-01, (float)+7.48508315e-01, (float)+3.93036469e-01, },
    {(float)+4.98493976e-01, (float)+9.97481587e-01, (float)+7.60527848e-01, (float)+7.48254412e-01, (float)+3.92177590e-01, },
    {(float)+4.97798888e-01, (float)+9.97505346e-01, (float)+7.59557539e-01, (float)+7.47873556e-01, (float)+3.91318710e-01, },
    {(float)+4.97103800e-01, (float)+9.97505346e-01, (float)+7.58781292e-01, (float)+7.47619652e-01, (float)+3.90459831e-01, },
    {(float)+4.96408712e-01, (float)+9.97505346e-01, (float)+7.58005046e-01, (float)+7.47365748e-01, (float)+3.89584434e-01, },
    {(float)+4.95713624e-01, (float)+9.97529104e-01, (float)+7.57228799e-01, (float)+7.46984893e-01, (float)+3.88725555e-01, },
    {(float)+4.95018536e-01, (float)+9.97529104e-01, (float)+7.56452552e-01, (float)+7.46730989e-01, (float)+3.87866675e-01, },
    {(float)+4.94323448e-01, (float)+9.97529104e-01, (float)+7.55482243e-01, (float)+7.46350133e-01, (float)+3.87007796e-01, },
    {(float)+4.93628360e-01, (float)+9.97552863e-01, (float)+7.54705997e-01, (float)+7.46096230e-01, (float)+3.86148916e-01, },
    {(float)+4.92933272e-01, (float)+9.97552863e-01, (float)+7.53929750e-01, (float)+7.45842326e-01, (float)+3.85290037e-01, },
    {(float)+4.92238184e-01, (float)+9.97552863e-01, (float)+7.53153503e-01, (float)+7.45461470e-01, (float)+3.84431158e-01, },
    {(float)+4.91543095e-01, (float)+9.97576622e-01, (float)+7.52377256e-01, (float)+7.45207566e-01, (float)+3.83555761e-01, },
    {(float)+4.90848007e-01, (float)+9.97576622e-01, (float)+7.51406947e-01, (float)+7.44826711e-01, (float)+3.82696882e-01, },
    {(float)+4.90152919e-01, (float)+9.97576622e-01, (float)+7.50630701e-01, (float)+7.44572807e-01, (float)+3.81838002e-01, },
    {(float)+4.89457831e-01, (float)+9.97600380e-01, (float)+7.49854454e-01, (float)+7.44318903e-01, (float)+3.80979123e-01, },
    {(float)+4.88762743e-01, (float)+9.97600380e-01, (float)+7.49078207e-01, (float)+7.43938047e-01, (float)+3.80120243e-01, },
    {(float)+4.88067655e-01, (float)+9.97624139e-01, (float)+7.48301960e-01, (float)+7.43684144e-01, (float)+3.79261364e-01, },
    {(float)+4.87372567e-01, (float)+9.97624139e-01, (float)+7.47525713e-01, (float)+7.43430240e-01, (float)+3.78402484e-01, },
    {(float)+4.86677479e-01, (float)+9.97624139e-01, (float)+7.46555405e-01, (float)+7.43049384e-01, (float)+3.77543605e-01, },
    {(float)+4.85982391e-01, (float)+9.97647897e-01, (float)+7.45779158e-01, (float)+7.42795481e-01, (float)+3.76668208e-01, },
    {(float)+4.85287303e-01, (float)+9.97647897e-01, (float)+7.45002911e-01, (float)+7.42414625e-01, (float)+3.75809329e-01, },
    {(float)+4.84592215e-01, (float)+9.97647897e-01, (float)+7.44226664e-01, (float)+7.42160721e-01, (float)+3.74950449e-01, },
    {(float)+4.83897127e-01, (float)+9.97671656e-01, (float)+7.43450417e-01, (float)+7.41906817e-01, (float)+3.74091570e-01, },
    {(float)+4.83202039e-01, (float)+9.97671656e-01, (float)+7.42480109e-01, (float)+7.41525962e-01, (float)+3.73232690e-01, },
    {(float)+4.82506951e-01, (float)+9.97671656e-01, (float)+7.41703862e-01, (float)+7.41272058e-01, (float)+3.72373811e-01, },
    {(float)+4.81811863e-01, (float)+9.97695415e-01, (float)+7.40927615e-01, (float)+7.40891202e-01, (float)+3.71514931e-01, },
    {(float)+4.81116775e-01, (float)+9.97695415e-01, (float)+7.40151368e-01, (float)+7.40637298e-01, (float)+3.70639535e-01, },
    {(float)+4.80421687e-01, (float)+9.97695415e-01, (float)+7.39375121e-01, (float)+7.40383395e-01, (float)+3.69780655e-01, },
    {(float)+4.79726599e-01, (float)+9.97719173e-01, (float)+7.38404813e-01, (float)+7.40002539e-01, (float)+3.68921776e-01, },
    {(float)+4.79031511e-01, (float)+9.97719173e-01, (float)+7.37628566e-01, (float)+7.39748635e-01, (float)+3.68062896e-01, },
    {(float)+4.78336423e-01, (float)+9.97719173e-01, (float)+7.36852319e-01, (float)+7.39494731e-01, (float)+3.67204017e-01, },
    {(float)+4.77641335e-01, (float)+9.97742932e-01, (float)+7.36076072e-01, (float)+7.39113876e-01, (float)+3.66345137e-01, },
    {(float)+4.76946247e-01, (float)+9.97742932e-01, (float)+7.35299825e-01, (float)+7.38859972e-01, (float)+3.65486258e-01, },
    {(float)+4.76251158e-01, (float)+9.97742932e-01, (float)+7.34329517e-01, (float)+7.38479116e-01, (float)+3.64610862e-01, },
    {(float)+4.75556070e-01, (float)+9.97766690e-01, (float)+7.33553270e-01, (float)+7.38225213e-01, (float)+3.63751982e-01, },
    {(float)+4.74860982e-01, (float)+9.97766690e-01, (float)+7.32777023e-01, (float)+7.37971309e-01, (float)+3.62893103e-01, },
    {(float)+4.74165894e-01, (float)+9.97766690e-01, (float)+7.32000776e-01, (float)+7.37590453e-01, (float)+3.62034223e-01, },
    {(float)+4.73470806e-01, (float)+9.97790449e-01, (float)+7.31224529e-01, (float)+7.37336549e-01, (float)+3.61175344e-01, },
    {(float)+4.72775718e-01, (float)+9.97790449e-01, (float)+7.30254221e-01, (float)+7.37082646e-01, (float)+3.60316464e-01, },
    {(float)+4.72080630e-01, (float)+9.97790449e-01, (float)+7.29477974e-01, (float)+7.36701790e-01, (float)+3.59457585e-01, },
    {(float)+4.71385542e-01, (float)+9.97814208e-01, (float)+7.28701727e-01, (float)+7.36447886e-01, (float)+3.58582188e-01, },
    {(float)+4.70690454e-01, (float)+9.97814208e-01, (float)+7.27925480e-01, (float)+7.36067031e-01, (float)+3.57723309e-01, },
    {(float)+4.69995366e-01, (float)+9.97814208e-01, (float)+7.27149233e-01, (float)+7.35813127e-01, (float)+3.56864429e-01, },
    {(float)+4.69300278e-01, (float)+9.97837966e-01, (float)+7.26178925e-01, (float)+7.35559223e-01, (float)+3.56005550e-01, },
    {(float)+4.68605190e-01, (float)+9.97837966e-01, (float)+7.25402678e-01, (float)+7.35178367e-01, (float)+3.55146670e-01, },
    {(float)+4.67910102e-01, (float)+9.97837966e-01, (float)+7.24626431e-01, (float)+7.34924464e-01, (float)+3.54287791e-01, },
    {(float)+4.67215014e-01, (float)+9.97861725e-01, (float)+7.23850184e-01, (float)+7.34543608e-01, (float)+3.53428911e-01, },
    {(float)+4.66519926e-01, (float)+9.97861725e-01, (float)+7.23073938e-01, (float)+7.34289704e-01, (float)+3.52553515e-01, },
    {(float)+4.65824838e-01, (float)+9.97861725e-01, (float)+7.22103629e-01, (float)+7.34035800e-01, (float)+3.51694635e-01, },
    {(float)+4.65129750e-01, (float)+9.97885483e-01, (float)+7.21327382e-01, (float)+7.33654945e-01, (float)+3.50835756e-01, },
    {(float)+4.64434662e-01, (float)+9.97885483e-01, (float)+7.20551135e-01, (float)+7.33401041e-01, (float)+3.49976876e-01, },
    {(float)+4.63739574e-01, (float)+9.97885483e-01, (float)+7.19774888e-01, (float)+7.33147137e-01, (float)+3.49117997e-01, },
    {(float)+4.62928638e-01, (float)+9.97909242e-01, (float)+7.18998642e-01, (float)+7.32766282e-01, (float)+3.48259117e-01, },
    {(float)+4.62233550e-01, (float)+9.97909242e-01, (float)+7.18028333e-01, (float)+7.32512378e-01, (float)+3.47400238e-01, },
    {(float)+4.61538462e-01, (float)+9.97909242e-01, (float)+7.17252086e-01, (float)+7.32131522e-01, (float)+3.46524841e-01, },
    {(float)+4.60843373e-01, (float)+9.97933001e-01, (float)+7.16475839e-01, (float)+7.31877618e-01, (float)+3.45665962e-01, },
    {(float)+4.60148285e-01, (float)+9.97933001e-01, (float)+7.15699592e-01, (float)+7.31623715e-01, (float)+3.44807082e-01, },
    {(float)+4.59453197e-01, (float)+9.97933001e-01, (float)+7.14923346e-01, (float)+7.31242859e-01, (float)+3.43948203e-01, },
    {(float)+4.58758109e-01, (float)+9.97956759e-01, (float)+7.13953037e-01, (float)+7.30988955e-01, (float)+3.43089323e-01, },
    {(float)+4.58063021e-01, (float)+9.97956759e-01, (float)+7.13176790e-01, (float)+7.30608100e-01, (float)+3.42230444e-01, },
    {(float)+4.57367933e-01, (float)+9.97956759e-01, (float)+7.12400543e-01, (float)+7.30354196e-01, (float)+3.41371564e-01, },
    {(float)+4.56672845e-01, (float)+9.97980518e-01, (float)+7.11624297e-01, (float)+7.30100292e-01, (float)+3.40496168e-01, },
    {(float)+4.55977757e-01, (float)+9.97980518e-01, (float)+7.10848050e-01, (float)+7.29719436e-01, (float)+3.39637289e-01, },
    {(float)+4.55282669e-01, (float)+9.97980518e-01, (float)+7.09877741e-01, (float)+7.29465533e-01, (float)+3.38778409e-01, },
    {(float)+4.54587581e-01, (float)+9.98004277e-01, (float)+7.09101494e-01, (float)+7.29211629e-01, (float)+3.37919530e-01, },
    {(float)+4.53892493e-01, (float)+9.98004277e-01, (float)+7.08325247e-01, (float)+7.28830773e-01, (float)+3.37060650e-01, },
    {(float)+4.53197405e-01, (float)+9.98004277e-01, (float)+7.07549001e-01, (float)+7.28576869e-01, (float)+3.36201771e-01, },
    {(float)+4.52502317e-01, (float)+9.98028035e-01, (float)+7.06772754e-01, (float)+7.28196014e-01, (float)+3.35342891e-01, },
    {(float)+4.51807229e-01, (float)+9.98028035e-01, (float)+7.05802445e-01, (float)+7.27942110e-01, (float)+3.34467495e-01, },
    {(float)+4.51112141e-01, (float)+9.98028035e-01, (float)+7.05026198e-01, (float)+7.27688206e-01, (float)+3.33608615e-01, },
    {(float)+4.50417053e-01, (float)+9.98051794e-01, (float)+7.04249951e-01, (float)+7.27307351e-01, (float)+3.32749736e-01, },
    {(float)+4.49721965e-01, (float)+9.98051794e-01, (float)+7.03473705e-01, (float)+7.27053447e-01, (float)+3.31890856e-01, },
    {(float)+4.49026877e-01, (float)+9.98051794e-01, (float)+7.02697458e-01, (float)+7.26799543e-01, (float)+3.31031977e-01, },
    {(float)+4.48331789e-01, (float)+9.98075552e-01, (float)+7.01727149e-01, (float)+7.26418687e-01, (float)+3.30173097e-01, },
    {(float)+4.47636701e-01, (float)+9.98075552e-01, (float)+7.00950902e-01, (float)+7.26164784e-01, (float)+3.29314218e-01, },
    {(float)+4.46941613e-01, (float)+9.98075552e-01, (float)+7.00174656e-01, (float)+7.25783928e-01, (float)+3.28455338e-01, },
    {(float)+4.46246525e-01, (float)+9.98099311e-01, (float)+6.99398409e-01, (float)+7.25530024e-01, (float)+3.27579942e-01, },
    {(float)+4.45551437e-01, (float)+9.98099311e-01, (float)+6.98622162e-01, (float)+7.25276120e-01, (float)+3.26721062e-01, },
    {(float)+4.44856348e-01, (float)+9.98099311e-01, (float)+6.97651853e-01, (float)+7.24895265e-01, (float)+3.25862183e-01, },
    {(float)+4.44161260e-01, (float)+9.98123070e-01, (float)+6.96875606e-01, (float)+7.24641361e-01, (float)+3.25003303e-01, },
    {(float)+4.43466172e-01, (float)+9.98123070e-01, (float)+6.96099360e-01, (float)+7.24260505e-01, (float)+3.24144424e-01, },
    {(float)+4.42771084e-01, (float)+9.98123070e-01, (float)+6.95323113e-01, (float)+7.24006601e-01, (float)+3.23285544e-01, },
    {(float)+4.42075996e-01, (float)+9.98146828e-01, (float)+6.94546866e-01, (float)+7.23752698e-01, (float)+3.22426665e-01, },
    {(float)+4.41380908e-01, (float)+9.98146828e-01, (float)+6.93770619e-01, (float)+7.23371842e-01, (float)+3.21551268e-01, },
    {(float)+4.40685820e-01, (float)+9.98146828e-01, (float)+6.92800310e-01, (float)+7.23117938e-01, (float)+3.20692389e-01, },
    {(float)+4.39990732e-01, (float)+9.98170587e-01, (float)+6.92024064e-01, (float)+7.22864035e-01, (float)+3.19833510e-01, },
    {(float)+4.39295644e-01, (float)+9.98170587e-01, (float)+6.91247817e-01, (float)+7.22483179e-01, (float)+3.18974630e-01, },
    {(float)+4.38600556e-01, (float)+9.98170587e-01, (float)+6.90471570e-01, (float)+7.22229275e-01, (float)+3.18115751e-01, },
    {(float)+4.37905468e-01, (float)+9.98194345e-01, (float)+6.89695323e-01, (float)+7.21848419e-01, (float)+3.17256871e-01, },
    {(float)+4.37210380e-01, (float)+9.98194345e-01, (float)+6.88725015e-01, (float)+7.21594516e-01, (float)+3.16397992e-01, },
    {(float)+4.36515292e-01, (float)+9.98194345e-01, (float)+6.87948768e-01, (float)+7.21340612e-01, (float)+3.15522595e-01, },
    {(float)+4.35820204e-01, (float)+9.98218104e-01, (float)+6.87172521e-01, (float)+7.20959756e-01, (float)+3.14663716e-01, },
    {(float)+4.35125116e-01, (float)+9.98218104e-01, (float)+6.86396274e-01, (float)+7.20705852e-01, (float)+3.13804836e-01, },
    {(float)+4.34430028e-01, (float)+9.98218104e-01, (float)+6.85620027e-01, (float)+7.20324997e-01, (float)+3.12945957e-01, },
    {(float)+4.33734940e-01, (float)+9.98241863e-01, (float)+6.84649719e-01, (float)+7.20071093e-01, (float)+3.12087077e-01, },
    {(float)+4.33039852e-01, (float)+9.98241863e-01, (float)+6.83873472e-01, (float)+7.19817189e-01, (float)+3.11228198e-01, },
    {(float)+4.32344764e-01, (float)+9.98241863e-01, (float)+6.83097225e-01, (float)+7.19436334e-01, (float)+3.10369318e-01, },
    {(float)+4.31649676e-01, (float)+9.98265621e-01, (float)+6.82320978e-01, (float)+7.19182430e-01, (float)+3.09493922e-01, },
    {(float)+4.30954588e-01, (float)+9.98265621e-01, (float)+6.81544731e-01, (float)+7.18928526e-01, (float)+3.08635042e-01, },
    {(float)+4.30259500e-01, (float)+9.98265621e-01, (float)+6.80574423e-01, (float)+7.18547670e-01, (float)+3.07776163e-01, },
    {(float)+4.29564411e-01, (float)+9.98289380e-01, (float)+6.79798176e-01, (float)+7.18293767e-01, (float)+3.06917283e-01, },
    {(float)+4.28869323e-01, (float)+9.98289380e-01, (float)+6.79021929e-01, (float)+7.17912911e-01, (float)+3.06058404e-01, },
    {(float)+4.28174235e-01, (float)+9.98313139e-01, (float)+6.78245682e-01, (float)+7.17659007e-01, (float)+3.05199524e-01, },
    {(float)+4.27479147e-01, (float)+9.98313139e-01, (float)+6.77469435e-01, (float)+7.17405103e-01, (float)+3.04340645e-01, },
    {(float)+4.26784059e-01, (float)+9.98313139e-01, (float)+6.76499127e-01, (float)+7.17024248e-01, (float)+3.03465248e-01, },
    {(float)+4.26088971e-01, (float)+9.98336897e-01, (float)+6.75722880e-01, (float)+7.16770344e-01, (float)+3.02606369e-01, },
    {(float)+4.25393883e-01, (float)+9.98336897e-01, (float)+6.74946633e-01, (float)+7.16516440e-01, (float)+3.01747489e-01, },
    {(float)+4.24698795e-01, (float)+9.98336897e-01, (float)+6.74170386e-01, (float)+7.16135585e-01, (float)+3.00888610e-01, },
    {(float)+4.24003707e-01, (float)+9.98360656e-01, (float)+6.73394139e-01, (float)+7.15881681e-01, (float)+3.00029730e-01, },
    {(float)+4.23308619e-01, (float)+9.98360656e-01, (float)+6.72423831e-01, (float)+7.15500825e-01, (float)+2.99170851e-01, },
    {(float)+4.22613531e-01, (float)+9.98360656e-01, (float)+6.71647584e-01, (float)+7.15246921e-01, (float)+2.98311971e-01, },
    {(float)+4.21918443e-01, (float)+9.98384414e-01, (float)+6.70871337e-01, (float)+7.14993018e-01, (float)+2.97436575e-01, },
    {(float)+4.21223355e-01, (float)+9.98384414e-01, (float)+6.70095090e-01, (float)+7.14612162e-01, (float)+2.96577696e-01, },
    {(float)+4.20528267e-01, (float)+9.98384414e-01, (float)+6.69318843e-01, (float)+7.14358258e-01, (float)+2.95718816e-01, },
    {(float)+4.19833179e-01, (float)+9.98408173e-01, (float)+6.68348535e-01, (float)+7.13977403e-01, (float)+2.94859937e-01, },
    {(float)+4.19138091e-01, (float)+9.98408173e-01, (float)+6.67572288e-01, (float)+7.13723499e-01, (float)+2.94001057e-01, },
    {(float)+4.18443003e-01, (float)+9.98408173e-01, (float)+6.66796041e-01, (float)+7.13469595e-01, (float)+2.93142178e-01, },
    {(float)+4.17747915e-01, (float)+9.98431932e-01, (float)+6.66019794e-01, (float)+7.13088739e-01, (float)+2.92283298e-01, },
    {(float)+4.17052827e-01, (float)+9.98431932e-01, (float)+6.65243547e-01, (float)+7.12834836e-01, (float)+2.91407902e-01, },
    {(float)+4.16357739e-01, (float)+9.98431932e-01, (float)+6.64273239e-01, (float)+7.12580932e-01, (float)+2.90549022e-01, },
    {(float)+4.15662651e-01, (float)+9.98455690e-01, (float)+6.63496992e-01, (float)+7.12200076e-01, (float)+2.89690143e-01, },
    {(float)+4.14967563e-01, (float)+9.98455690e-01, (float)+6.62720745e-01, (float)+7.11946172e-01, (float)+2.88831263e-01, },
    {(float)+4.14272475e-01, (float)+9.98455690e-01, (float)+6.61944498e-01, (float)+7.11565317e-01, (float)+2.87972384e-01, },
    {(float)+4.13577386e-01, (float)+9.98479449e-01, (float)+6.61168252e-01, (float)+7.11311413e-01, (float)+2.87113504e-01, },
    {(float)+4.12882298e-01, (float)+9.98479449e-01, (float)+6.60197943e-01, (float)+7.11057509e-01, (float)+2.86254625e-01, },
    {(float)+4.12187210e-01, (float)+9.98479449e-01, (float)+6.59421696e-01, (float)+7.10676654e-01, (float)+2.85395745e-01, },
    {(float)+4.11492122e-01, (float)+9.98503207e-01, (float)+6.58645449e-01, (float)+7.10422750e-01, (float)+2.84520349e-01, },
    {(float)+4.10681186e-01, (float)+9.98503207e-01, (float)+6.57869202e-01, (float)+7.10041894e-01, (float)+2.83661469e-01, },
    {(float)+4.09986098e-01, (float)+9.98503207e-01, (float)+6.57092956e-01, (float)+7.09787990e-01, (float)+2.82802590e-01, },
    {(float)+4.09291010e-01, (float)+9.98526966e-01, (float)+6.56122647e-01, (float)+7.09534087e-01, (float)+2.81943710e-01, },
    {(float)+4.08595922e-01, (float)+9.98526966e-01, (float)+6.55346400e-01, (float)+7.09153231e-01, (float)+2.81084831e-01, },
    {(float)+4.07900834e-01, (float)+9.98526966e-01, (float)+6.54570153e-01, (float)+7.08899327e-01, (float)+2.80225951e-01, },
    {(float)+4.07205746e-01, (float)+9.98550725e-01, (float)+6.53793906e-01, (float)+7.08645423e-01, (float)+2.79367072e-01, },
    {(float)+4.06510658e-01, (float)+9.98550725e-01, (float)+6.53017660e-01, (float)+7.08264568e-01, (float)+2.78491675e-01, },
    {(float)+4.05815570e-01, (float)+9.98550725e-01, (float)+6.52047351e-01, (float)+7.08010664e-01, (float)+2.77632796e-01, },
    {(float)+4.05120482e-01, (float)+9.98574483e-01, (float)+6.51271104e-01, (float)+7.07629808e-01, (float)+2.76773916e-01, },
    {(float)+4.04425394e-01, (float)+9.98574483e-01, (float)+6.50494857e-01, (float)+7.07375905e-01, (float)+2.75915037e-01, },
    {(float)+4.03730306e-01, (float)+9.98574483e-01, (float)+6.49718611e-01, (float)+7.07122001e-01, (float)+2.75056158e-01, },
    {(float)+4.03035218e-01, (float)+9.98598242e-01, (float)+6.48942364e-01, (float)+7.06741145e-01, (float)+2.74197278e-01, },
    {(float)+4.02340130e-01, (float)+9.98598242e-01, (float)+6.47972055e-01, (float)+7.06487241e-01, (float)+2.73338399e-01, },
    {(float)+4.01645042e-01, (float)+9.98598242e-01, (float)+6.47195808e-01, (float)+7.06233338e-01, (float)+2.72463002e-01, },
    {(float)+4.00949954e-01, (float)+9.98622000e-01, (float)+6.46419561e-01, (float)+7.05852482e-01, (float)+2.71604123e-01, },
    {(float)+4.00254866e-01, (float)+9.98622000e-01, (float)+6.45643315e-01, (float)+7.05598578e-01, (float)+2.70745243e-01, },
    {(float)+3.99559778e-01, (float)+9.98622000e-01, (float)+6.44867068e-01, (float)+7.05217722e-01, (float)+2.69886364e-01, },
    {(float)+3.98864690e-01, (float)+9.98645759e-01, (float)+6.43896759e-01, (float)+7.04963819e-01, (float)+2.69027484e-01, },
    {(float)+3.98169601e-01, (float)+9.98645759e-01, (float)+6.43120512e-01, (float)+7.04709915e-01, (float)+2.68168605e-01, },
    {(float)+3.97474513e-01, (float)+9.98645759e-01, (float)+6.42344265e-01, (float)+7.04329059e-01, (float)+2.67309725e-01, },
    {(float)+3.96779425e-01, (float)+9.98669518e-01, (float)+6.41568019e-01, (float)+7.04075156e-01, (float)+2.66434329e-01, },
    {(float)+3.96084337e-01, (float)+9.98669518e-01, (float)+6.40791772e-01, (float)+7.03694300e-01, (float)+2.65575449e-01, },
    {(float)+3.95389249e-01, (float)+9.98669518e-01, (float)+6.40015525e-01, (float)+7.03440396e-01, (float)+2.64716570e-01, },
    {(float)+3.94694161e-01, (float)+9.98693276e-01, (float)+6.39045216e-01, (float)+7.03186492e-01, (float)+2.63857690e-01, },
    {(float)+3.93999073e-01, (float)+9.98693276e-01, (float)+6.38268970e-01, (float)+7.02805637e-01, (float)+2.62998811e-01, },
    {(float)+3.93303985e-01, (float)+9.98693276e-01, (float)+6.37492723e-01, (float)+7.02551733e-01, (float)+2.62139931e-01, },
    {(float)+3.92608897e-01, (float)+9.98717035e-01, (float)+6.36716476e-01, (float)+7.02297829e-01, (float)+2.61281052e-01, },
    {(float)+3.91913809e-01, (float)+9.98717035e-01, (float)+6.35940229e-01, (float)+7.01916973e-01, (float)+2.60405655e-01, },
    {(float)+3.91218721e-01, (float)+9.98717035e-01, (float)+6.34969920e-01, (float)+7.01663070e-01, (float)+2.59546776e-01, },
    {(float)+3.90523633e-01, (float)+9.98740794e-01, (float)+6.34193674e-01, (float)+7.01282214e-01, (float)+2.58687896e-01, },
    {(float)+3.89828545e-01, (float)+9.98740794e-01, (float)+6.33417427e-01, (float)+7.01028310e-01, (float)+2.57829017e-01, },
    {(float)+3.89133457e-01, (float)+9.98740794e-01, (float)+6.32641180e-01, (float)+7.00774406e-01, (float)+2.56970137e-01, },
    {(float)+3.88438369e-01, (float)+9.98764552e-01, (float)+6.31864933e-01, (float)+7.00393551e-01, (float)+2.56111258e-01, },
    {(float)+3.87743281e-01, (float)+9.98764552e-01, (float)+6.30894624e-01, (float)+7.00139647e-01, (float)+2.55252378e-01, },
    {(float)+3.87048193e-01, (float)+9.98764552e-01, (float)+6.30118378e-01, (float)+6.99758791e-01, (float)+2.54376982e-01, },
    {(float)+3.86353105e-01, (float)+9.98788311e-01, (float)+6.29342131e-01, (float)+6.99504888e-01, (float)+2.53518103e-01, },
    {(float)+3.85658017e-01, (float)+9.98788311e-01, (float)+6.28565884e-01, (float)+6.99250984e-01, (float)+2.52659223e-01, },
    {(float)+3.84962929e-01, (float)+9.98788311e-01, (float)+6.27789637e-01, (float)+6.98870128e-01, (float)+2.51800344e-01, },
    {(float)+3.84267841e-01, (float)+9.98812069e-01, (float)+6.26819329e-01, (float)+6.98616224e-01, (float)+2.50941464e-01, },
    {(float)+3.83572753e-01, (float)+9.98812069e-01, (float)+6.26043082e-01, (float)+6.98362321e-01, (float)+2.50082585e-01, },
    {(float)+3.82877665e-01, (float)+9.98812069e-01, (float)+6.25266835e-01, (float)+6.97981465e-01, (float)+2.49223705e-01, },
    {(float)+3.82182576e-01, (float)+9.98835828e-01, (float)+6.24490588e-01, (float)+6.97727561e-01, (float)+2.48348309e-01, },
    {(float)+3.81487488e-01, (float)+9.98835828e-01, (float)+6.23714341e-01, (float)+6.97346706e-01, (float)+2.47489429e-01, },
    {(float)+3.80792400e-01, (float)+9.98835828e-01, (float)+6.22744033e-01, (float)+6.97092802e-01, (float)+2.46630550e-01, },
    {(float)+3.80097312e-01, (float)+9.98859587e-01, (float)+6.21967786e-01, (float)+6.96838898e-01, (float)+2.45771670e-01, },
    {(float)+3.79402224e-01, (float)+9.98859587e-01, (float)+6.21191539e-01, (float)+6.96458042e-01, (float)+2.44912791e-01, },
    {(float)+3.78707136e-01, (float)+9.98859587e-01, (float)+6.20415292e-01, (float)+6.96204139e-01, (float)+2.44053911e-01, },
    {(float)+3.78012048e-01, (float)+9.98883345e-01, (float)+6.19639045e-01, (float)+6.95950235e-01, (float)+2.43195032e-01, },
    {(float)+3.77316960e-01, (float)+9.98883345e-01, (float)+6.18668737e-01, (float)+6.95569379e-01, (float)+2.42319635e-01, },
    {(float)+3.76621872e-01, (float)+9.98883345e-01, (float)+6.17892490e-01, (float)+6.95315475e-01, (float)+2.41460756e-01, },
    {(float)+3.75926784e-01, (float)+9.98907104e-01, (float)+6.17116243e-01, (float)+6.94934620e-01, (float)+2.40601876e-01, },
    {(float)+3.75231696e-01, (float)+9.98907104e-01, (float)+6.16339996e-01, (float)+6.94680716e-01, (float)+2.39742997e-01, },
    {(float)+3.74536608e-01, (float)+9.98907104e-01, (float)+6.15563749e-01, (float)+6.94426812e-01, (float)+2.38884117e-01, },
    {(float)+3.73841520e-01, (float)+9.98930862e-01, (float)+6.14593441e-01, (float)+6.94045957e-01, (float)+2.38025238e-01, },
    {(float)+3.73146432e-01, (float)+9.98930862e-01, (float)+6.13817194e-01, (float)+6.93792053e-01, (float)+2.37166358e-01, },
    {(float)+3.72451344e-01, (float)+9.98930862e-01, (float)+6.13040947e-01, (float)+6.93411197e-01, (float)+2.36307479e-01, },
    {(float)+3.71756256e-01, (float)+9.98954621e-01, (float)+6.12264700e-01, (float)+6.93157293e-01, (float)+2.35432082e-01, },
    {(float)+3.71061168e-01, (float)+9.98954621e-01, (float)+6.11488453e-01, (float)+6.92903390e-01, (float)+2.34573203e-01, },
    {(float)+3.70366080e-01, (float)+9.98954621e-01, (float)+6.10518145e-01, (float)+6.92522534e-01, (float)+2.33714323e-01, },
    {(float)+3.69670992e-01, (float)+9.98978380e-01, (float)+6.09741898e-01, (float)+6.92268630e-01, (float)+2.32855444e-01, },
    {(float)+3.68975904e-01, (float)+9.98978380e-01, (float)+6.08965651e-01, (float)+6.92014726e-01, (float)+2.31996564e-01, },
    {(float)+3.68280816e-01, (float)+9.99002138e-01, (float)+6.08189404e-01, (float)+6.91633871e-01, (float)+2.31137685e-01, },
    {(float)+3.67585728e-01, (float)+9.99002138e-01, (float)+6.07413157e-01, (float)+6.91379967e-01, (float)+2.30278805e-01, },
    {(float)+3.66890639e-01, (float)+9.99002138e-01, (float)+6.06442849e-01, (float)+6.90999111e-01, (float)+2.29403409e-01, },
    {(float)+3.66195551e-01, (float)+9.99025897e-01, (float)+6.05666602e-01, (float)+6.90745208e-01, (float)+2.28544530e-01, },
    {(float)+3.65500463e-01, (float)+9.99025897e-01, (float)+6.04890355e-01, (float)+6.90491304e-01, (float)+2.27685650e-01, },
    {(float)+3.64805375e-01, (float)+9.99025897e-01, (float)+6.04114108e-01, (float)+6.90110448e-01, (float)+2.26826771e-01, },
    {(float)+3.64110287e-01, (float)+9.99049656e-01, (float)+6.03337861e-01, (float)+6.89856544e-01, (float)+2.25967891e-01, },
    {(float)+3.63415199e-01, (float)+9.99049656e-01, (float)+6.02367553e-01, (float)+6.89475689e-01, (float)+2.25109012e-01, },
    {(float)+3.62720111e-01, (float)+9.99049656e-01, (float)+6.01591306e-01, (float)+6.89221785e-01, (float)+2.24250132e-01, },
    {(float)+3.62025023e-01, (float)+9.99073414e-01, (float)+6.00815059e-01, (float)+6.88967881e-01, (float)+2.23374736e-01, },
    {(float)+3.61329935e-01, (float)+9.99073414e-01, (float)+6.00038812e-01, (float)+6.88587026e-01, (float)+2.22515856e-01, },
    {(float)+3.60634847e-01, (float)+9.99073414e-01, (float)+5.99262565e-01, (float)+6.88333122e-01, (float)+2.21656977e-01, },
    {(float)+3.59939759e-01, (float)+9.99097173e-01, (float)+5.98292257e-01, (float)+6.88079218e-01, (float)+2.20798097e-01, },
    {(float)+3.59244671e-01, (float)+9.99097173e-01, (float)+5.97516010e-01, (float)+6.87698362e-01, (float)+2.19939218e-01, },
    {(float)+3.58433735e-01, (float)+9.99097173e-01, (float)+5.96739763e-01, (float)+6.87444459e-01, (float)+2.19080338e-01, },
    {(float)+3.57738647e-01, (float)+9.99120931e-01, (float)+5.95963516e-01, (float)+6.87063603e-01, (float)+2.18221459e-01, },
    {(float)+3.57043559e-01, (float)+9.99120931e-01, (float)+5.95187270e-01, (float)+6.86809699e-01, (float)+2.17346062e-01, },
    {(float)+3.56348471e-01, (float)+9.99120931e-01, (float)+5.94216961e-01, (float)+6.86555795e-01, (float)+2.16487183e-01, },
    {(float)+3.55653383e-01, (float)+9.99144690e-01, (float)+5.93440714e-01, (float)+6.86174940e-01, (float)+2.15628303e-01, },
    {(float)+3.54958295e-01, (float)+9.99144690e-01, (float)+5.92664467e-01, (float)+6.85921036e-01, (float)+2.14769424e-01, },
    {(float)+3.54263207e-01, (float)+9.99144690e-01, (float)+5.91888220e-01, (float)+6.85667132e-01, (float)+2.13910544e-01, },
    {(float)+3.53568119e-01, (float)+9.99168449e-01, (float)+5.91111974e-01, (float)+6.85286277e-01, (float)+2.13051665e-01, },
    {(float)+3.52873031e-01, (float)+9.99168449e-01, (float)+5.90141665e-01, (float)+6.85032373e-01, (float)+2.12192785e-01, },
    {(float)+3.52177943e-01, (float)+9.99168449e-01, (float)+5.89365418e-01, (float)+6.84651517e-01, (float)+2.11317389e-01, },
    {(float)+3.51482854e-01, (float)+9.99192207e-01, (float)+5.88589171e-01, (float)+6.84397613e-01, (float)+2.10458510e-01, },
    {(float)+3.50787766e-01, (float)+9.99192207e-01, (float)+5.87812925e-01, (float)+6.84143710e-01, (float)+2.09599630e-01, },
    {(float)+3.50092678e-01, (float)+9.99192207e-01, (float)+5.87036678e-01, (float)+6.83762854e-01, (float)+2.08740751e-01, },
    {(float)+3.49397590e-01, (float)+9.99215966e-01, (float)+5.86260431e-01, (float)+6.83508950e-01, (float)+2.07881871e-01, },
    {(float)+3.48702502e-01, (float)+9.99215966e-01, (float)+5.85290122e-01, (float)+6.83128094e-01, (float)+2.07022992e-01, },
    {(float)+3.48007414e-01, (float)+9.99215966e-01, (float)+5.84513875e-01, (float)+6.82874191e-01, (float)+2.06164112e-01, },
    {(float)+3.47312326e-01, (float)+9.99239724e-01, (float)+5.83737629e-01, (float)+6.82620287e-01, (float)+2.05288716e-01, },
    {(float)+3.46617238e-01, (float)+9.99239724e-01, (float)+5.82961382e-01, (float)+6.82239431e-01, (float)+2.04429836e-01, },
    {(float)+3.45922150e-01, (float)+9.99239724e-01, (float)+5.82185135e-01, (float)+6.81985527e-01, (float)+2.03570957e-01, },
    {(float)+3.45227062e-01, (float)+9.99263483e-01, (float)+5.81214826e-01, (float)+6.81731624e-01, (float)+2.02712077e-01, },
    {(float)+3.44531974e-01, (float)+9.99263483e-01, (float)+5.80438579e-01, (float)+6.81350768e-01, (float)+2.01853198e-01, },
    };
    vector<vector<float>> transpose_result = transposeMatrix(initial_data);
    // cout << "转置结果：" << transpose_result.size() << " " << transpose_result[0].size() << endl;
    vector<vector<float>> input_data = transpose_result;

    if (input_data.size() != transpose_result.size())return -1;

    vector<vector<float>> target = transpose_result;

    int function_type = 1;
    int node = 30;//第一层节点
    int laye2_neural_nodeCount = 10;
    int epochs = 100;//训练轮次

    int target_mse = 0.000001;
    float lr = 0.05;

    // 定义第一层
    auto result_tuple_layer1 = initial_neurallayer_output(input_data, node, function_type);
    // 从 tuple 中获取返回的值
    vector<vector<float>> layer1_output = get<0>(result_tuple_layer1);
    vector<float> layer1_biasmatrix = get<1>(result_tuple_layer1);
    vector<vector<float>> layel1_weightmatrix = get<2>(result_tuple_layer1);

    vector<vector<float>> layer1_output_ini(layer1_output.size(), vector<float>(layer1_output[0].size(), 0));


    // 定义第二层
    auto result_tuple_layer2 = initial_neurallayer_output(layer1_output, laye2_neural_nodeCount, function_type);
    vector<vector<float>> layer2_output = get<0>(result_tuple_layer2);
    vector<float> layer2_biasmatrix = get<1>(result_tuple_layer2);
    vector<vector<float>> layel2_weightmatrix = get<2>(result_tuple_layer2);

    vector<vector<float>> layer2_output_ini(layer2_output.size(), vector<float>(layer2_output[0].size(), 0));

    // 输出层
    auto result_tuple_op = initial_neurallayer_output(layer2_output, target.size(), function_type);
    vector<vector<float>> output = get<0>(result_tuple_op);
    vector<float> op_biasmatrix = get<1>(result_tuple_op);
    vector<vector<float>> op_weightmatrix = get<2>(result_tuple_op);
    vector<vector<float>> error = calculateerror(output, target);

    float mse = calculateMSE(error);
    vector<vector<float>> layer2_error = calculateerror(layer2_output, layer2_output_ini);
    vector<vector<float>> layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出

    //神经网络训练主循环
    for (int epoch = 0; epoch < epochs; epoch++) {
        layer1_output = training_forward_neurallayer_output(input_data, layel1_weightmatrix, layer1_biasmatrix, node, function_type);

        // 第二层
        layer2_output = training_forward_neurallayer_output(layer1_output, layel2_weightmatrix, layer2_biasmatrix, laye2_neural_nodeCount, function_type);

        // 输出层
        output = training_forward_neurallayer_output(layer2_output, op_weightmatrix, op_biasmatrix, target.size(), function_type);

        error = calculateerror(output, target);
        mse = calculateMSE(error);
        layer2_error = calculateerror(layer2_output, layer2_output_ini);
        layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出
        update_weights_bias(op_weightmatrix, op_biasmatrix, output, error, layer2_output, lr);//参数顺序：该层权重，该层输出，该层误差，该层输入，学习率

        update_weights_bias(layel2_weightmatrix, layer2_biasmatrix,layer2_output, layer2_error, layer1_output, lr);
        update_weights_bias(layel1_weightmatrix, layer1_biasmatrix, layer1_output, layer1_error, input_data, lr);
        cout << "训练步数：" << epoch << "  " << "MSE:" << mse << endl;
        //lr = lr * exp(-0.05*epoch);效果不佳
        if (mse < target_mse) break;

    }
    //cout << "最后一次迭代输出结果：" << endl;
    //print2DArray(output);

    return 0;
}

*/







/*
int main()
{

    vector<vector<float>> initial_data= {
    {(float)+6.92191844e-01, (float)+9.95272036e-01, (float)+9.86803804e-01, (float)+8.32296560e-01, (float)+6.31590248e-01, },
    {(float)+6.91496756e-01, (float)+9.95272036e-01, (float)+9.86027557e-01, (float)+8.32042656e-01, (float)+6.30731369e-01, },
    {(float)+6.90801668e-01, (float)+9.95272036e-01, (float)+9.85251310e-01, (float)+8.31661800e-01, (float)+6.29872489e-01, },
    {(float)+6.90106580e-01, (float)+9.95295795e-01, (float)+9.84475063e-01, (float)+8.31407896e-01, (float)+6.28997093e-01, },
    {(float)+6.89411492e-01, (float)+9.95295795e-01, (float)+9.83698816e-01, (float)+8.31027041e-01, (float)+6.28138214e-01, },
    {(float)+6.88716404e-01, (float)+9.95295795e-01, (float)+9.82728508e-01, (float)+8.30773137e-01, (float)+6.27279334e-01, },
    {(float)+6.88021316e-01, (float)+9.95319553e-01, (float)+9.81952261e-01, (float)+8.30519233e-01, (float)+6.26420455e-01, },
    {(float)+6.87326228e-01, (float)+9.95319553e-01, (float)+9.81176014e-01, (float)+8.30138378e-01, (float)+6.25561575e-01, },
    {(float)+6.86631140e-01, (float)+9.95319553e-01, (float)+9.80399767e-01, (float)+8.29884474e-01, (float)+6.24702696e-01, },
    {(float)+6.85936052e-01, (float)+9.95343312e-01, (float)+9.79623520e-01, (float)+8.29630570e-01, (float)+6.23843816e-01, },
    {(float)+6.85240964e-01, (float)+9.95343312e-01, (float)+9.78653212e-01, (float)+8.29249714e-01, (float)+6.22968420e-01, },
    {(float)+6.84545876e-01, (float)+9.95343312e-01, (float)+9.77876965e-01, (float)+8.28995811e-01, (float)+6.22109540e-01, },
    {(float)+6.83850788e-01, (float)+9.95367071e-01, (float)+9.77100718e-01, (float)+8.28614955e-01, (float)+6.21250661e-01, },
    {(float)+6.83155700e-01, (float)+9.95367071e-01, (float)+9.76324471e-01, (float)+8.28361051e-01, (float)+6.20391781e-01, },
    {(float)+6.82460612e-01, (float)+9.95367071e-01, (float)+9.75548224e-01, (float)+8.28107147e-01, (float)+6.19532902e-01, },
    {(float)+6.81765524e-01, (float)+9.95390829e-01, (float)+9.74577916e-01, (float)+8.27726292e-01, (float)+6.18674022e-01, },
    {(float)+6.81070436e-01, (float)+9.95390829e-01, (float)+9.73801669e-01, (float)+8.27472388e-01, (float)+6.17815143e-01, },
    {(float)+6.80375348e-01, (float)+9.95390829e-01, (float)+9.73025422e-01, (float)+8.27091532e-01, (float)+6.16939746e-01, },
    {(float)+6.79680259e-01, (float)+9.95414588e-01, (float)+9.72249175e-01, (float)+8.26837629e-01, (float)+6.16080867e-01, },
    {(float)+6.78985171e-01, (float)+9.95414588e-01, (float)+9.71472928e-01, (float)+8.26583725e-01, (float)+6.15221987e-01, },
    {(float)+6.78290083e-01, (float)+9.95414588e-01, (float)+9.70502620e-01, (float)+8.26202869e-01, (float)+6.14363108e-01, },
    {(float)+6.77594995e-01, (float)+9.95438346e-01, (float)+9.69726373e-01, (float)+8.25948965e-01, (float)+6.13504228e-01, },
    {(float)+6.76899907e-01, (float)+9.95438346e-01, (float)+9.68950126e-01, (float)+8.25695062e-01, (float)+6.12645349e-01, },
    {(float)+6.76204819e-01, (float)+9.95438346e-01, (float)+9.68173879e-01, (float)+8.25314206e-01, (float)+6.11786469e-01, },
    {(float)+6.75509731e-01, (float)+9.95462105e-01, (float)+9.67397632e-01, (float)+8.25060302e-01, (float)+6.10927590e-01, },
    {(float)+6.74814643e-01, (float)+9.95462105e-01, (float)+9.66427324e-01, (float)+8.24679446e-01, (float)+6.10052193e-01, },
    {(float)+6.74119555e-01, (float)+9.95462105e-01, (float)+9.65651077e-01, (float)+8.24425543e-01, (float)+6.09193314e-01, },
    {(float)+6.73424467e-01, (float)+9.95485864e-01, (float)+9.64874830e-01, (float)+8.24171639e-01, (float)+6.08334434e-01, },
    {(float)+6.72729379e-01, (float)+9.95485864e-01, (float)+9.64098583e-01, (float)+8.23790783e-01, (float)+6.07475555e-01, },
    {(float)+6.72034291e-01, (float)+9.95485864e-01, (float)+9.63322337e-01, (float)+8.23536880e-01, (float)+6.06616675e-01, },
    {(float)+6.71223355e-01, (float)+9.95509622e-01, (float)+9.62546090e-01, (float)+8.23156024e-01, (float)+6.05757796e-01, },
    {(float)+6.70528267e-01, (float)+9.95509622e-01, (float)+9.61575781e-01, (float)+8.22902120e-01, (float)+6.04898916e-01, },
    {(float)+6.69833179e-01, (float)+9.95509622e-01, (float)+9.60799534e-01, (float)+8.22648216e-01, (float)+6.04023520e-01, },
    {(float)+6.69138091e-01, (float)+9.95533381e-01, (float)+9.60023287e-01, (float)+8.22267361e-01, (float)+6.03164641e-01, },
    {(float)+6.68443003e-01, (float)+9.95533381e-01, (float)+9.59247041e-01, (float)+8.22013457e-01, (float)+6.02305761e-01, },
    {(float)+6.67747915e-01, (float)+9.95557139e-01, (float)+9.58470794e-01, (float)+8.21759553e-01, (float)+6.01446882e-01, },
    {(float)+6.67052827e-01, (float)+9.95557139e-01, (float)+9.57500485e-01, (float)+8.21378697e-01, (float)+6.00588002e-01, },
    {(float)+6.66357739e-01, (float)+9.95557139e-01, (float)+9.56724238e-01, (float)+8.21124794e-01, (float)+5.99729123e-01, },
    {(float)+6.65662651e-01, (float)+9.95580898e-01, (float)+9.55947991e-01, (float)+8.20743938e-01, (float)+5.98870243e-01, },
    {(float)+6.64967563e-01, (float)+9.95580898e-01, (float)+9.55171745e-01, (float)+8.20490034e-01, (float)+5.97994847e-01, },
    {(float)+6.64272475e-01, (float)+9.95580898e-01, (float)+9.54395498e-01, (float)+8.20236131e-01, (float)+5.97135967e-01, },
    {(float)+6.63577386e-01, (float)+9.95604657e-01, (float)+9.53425189e-01, (float)+8.19855275e-01, (float)+5.96277088e-01, },
    {(float)+6.62882298e-01, (float)+9.95604657e-01, (float)+9.52648942e-01, (float)+8.19601371e-01, (float)+5.95418208e-01, },
    {(float)+6.62187210e-01, (float)+9.95604657e-01, (float)+9.51872696e-01, (float)+8.19347467e-01, (float)+5.94559329e-01, },
    {(float)+6.61492122e-01, (float)+9.95628415e-01, (float)+9.51096449e-01, (float)+8.18966612e-01, (float)+5.93700449e-01, },
    {(float)+6.60797034e-01, (float)+9.95628415e-01, (float)+9.50320202e-01, (float)+8.18712708e-01, (float)+5.92841570e-01, },
    {(float)+6.60101946e-01, (float)+9.95628415e-01, (float)+9.49349893e-01, (float)+8.18331852e-01, (float)+5.91966173e-01, },
    {(float)+6.59406858e-01, (float)+9.95652174e-01, (float)+9.48573646e-01, (float)+8.18077948e-01, (float)+5.91107294e-01, },
    {(float)+6.58711770e-01, (float)+9.95652174e-01, (float)+9.47797400e-01, (float)+8.17824045e-01, (float)+5.90248414e-01, },
    {(float)+6.58016682e-01, (float)+9.95652174e-01, (float)+9.47021153e-01, (float)+8.17443189e-01, (float)+5.89389535e-01, },
    {(float)+6.57321594e-01, (float)+9.95675933e-01, (float)+9.46244906e-01, (float)+8.17189285e-01, (float)+5.88530655e-01, },
    {(float)+6.56626506e-01, (float)+9.95675933e-01, (float)+9.45274597e-01, (float)+8.16808430e-01, (float)+5.87671776e-01, },
    {(float)+6.55931418e-01, (float)+9.95675933e-01, (float)+9.44498350e-01, (float)+8.16554526e-01, (float)+5.86812896e-01, },
    {(float)+6.55236330e-01, (float)+9.95699691e-01, (float)+9.43722104e-01, (float)+8.16300622e-01, (float)+5.85937500e-01, },
    {(float)+6.54541242e-01, (float)+9.95699691e-01, (float)+9.42945857e-01, (float)+8.15919766e-01, (float)+5.85078621e-01, },
    {(float)+6.53846154e-01, (float)+9.95699691e-01, (float)+9.42169610e-01, (float)+8.15665863e-01, (float)+5.84219741e-01, },
    {(float)+6.53151066e-01, (float)+9.95723450e-01, (float)+9.41199301e-01, (float)+8.15411959e-01, (float)+5.83360862e-01, },
    {(float)+6.52455978e-01, (float)+9.95723450e-01, (float)+9.40423055e-01, (float)+8.15031103e-01, (float)+5.82501982e-01, },
    {(float)+6.51760890e-01, (float)+9.95723450e-01, (float)+9.39646808e-01, (float)+8.14777199e-01, (float)+5.81643103e-01, },
    {(float)+6.51065802e-01, (float)+9.95747208e-01, (float)+9.38870561e-01, (float)+8.14396344e-01, (float)+5.80784223e-01, },
    {(float)+6.50370714e-01, (float)+9.95747208e-01, (float)+9.38094314e-01, (float)+8.14142440e-01, (float)+5.79908827e-01, },
    {(float)+6.49675626e-01, (float)+9.95747208e-01, (float)+9.37124005e-01, (float)+8.13888536e-01, (float)+5.79049947e-01, },
    {(float)+6.48980538e-01, (float)+9.95770967e-01, (float)+9.36347759e-01, (float)+8.13507681e-01, (float)+5.78191068e-01, },
    {(float)+6.48285449e-01, (float)+9.95770967e-01, (float)+9.35571512e-01, (float)+8.13253777e-01, (float)+5.77332188e-01, },
    {(float)+6.47590361e-01, (float)+9.95770967e-01, (float)+9.34795265e-01, (float)+8.12872921e-01, (float)+5.76473309e-01, },
    {(float)+6.46895273e-01, (float)+9.95794726e-01, (float)+9.34019018e-01, (float)+8.12619017e-01, (float)+5.75614429e-01, },
    {(float)+6.46200185e-01, (float)+9.95794726e-01, (float)+9.33048709e-01, (float)+8.12365114e-01, (float)+5.74755550e-01, },
    {(float)+6.45505097e-01, (float)+9.95794726e-01, (float)+9.32272463e-01, (float)+8.11984258e-01, (float)+5.73880153e-01, },
    {(float)+6.44810009e-01, (float)+9.95818484e-01, (float)+9.31496216e-01, (float)+8.11730354e-01, (float)+5.73021274e-01, },
    {(float)+6.44114921e-01, (float)+9.95818484e-01, (float)+9.30719969e-01, (float)+8.11476450e-01, (float)+5.72162394e-01, },
    {(float)+6.43419833e-01, (float)+9.95818484e-01, (float)+9.29943722e-01, (float)+8.11095595e-01, (float)+5.71303515e-01, },
    {(float)+6.42724745e-01, (float)+9.95842243e-01, (float)+9.28973414e-01, (float)+8.10841691e-01, (float)+5.70444635e-01, },
    {(float)+6.42029657e-01, (float)+9.95842243e-01, (float)+9.28197167e-01, (float)+8.10460835e-01, (float)+5.69585756e-01, },
    {(float)+6.41334569e-01, (float)+9.95842243e-01, (float)+9.27420920e-01, (float)+8.10206932e-01, (float)+5.68726876e-01, },
    {(float)+6.40639481e-01, (float)+9.95866001e-01, (float)+9.26644673e-01, (float)+8.09953028e-01, (float)+5.67867997e-01, },
    {(float)+6.39944393e-01, (float)+9.95866001e-01, (float)+9.25868426e-01, (float)+8.09572172e-01, (float)+5.66992600e-01, },
    {(float)+6.39249305e-01, (float)+9.95866001e-01, (float)+9.24898118e-01, (float)+8.09318268e-01, (float)+5.66133721e-01, },
    {(float)+6.38554217e-01, (float)+9.95889760e-01, (float)+9.24121871e-01, (float)+8.09064365e-01, (float)+5.65274841e-01, },
    {(float)+6.37859129e-01, (float)+9.95889760e-01, (float)+9.23345624e-01, (float)+8.08683509e-01, (float)+5.64415962e-01, },
    {(float)+6.37164041e-01, (float)+9.95889760e-01, (float)+9.22569377e-01, (float)+8.08429605e-01, (float)+5.63557082e-01, },
    {(float)+6.36468953e-01, (float)+9.95913519e-01, (float)+9.21793130e-01, (float)+8.08048750e-01, (float)+5.62698203e-01, },
    {(float)+6.35773865e-01, (float)+9.95913519e-01, (float)+9.20822822e-01, (float)+8.07794846e-01, (float)+5.61839323e-01, },
    {(float)+6.35078777e-01, (float)+9.95913519e-01, (float)+9.20046575e-01, (float)+8.07540942e-01, (float)+5.60963927e-01, },
    {(float)+6.34383689e-01, (float)+9.95937277e-01, (float)+9.19270328e-01, (float)+8.07160086e-01, (float)+5.60105048e-01, },
    {(float)+6.33688601e-01, (float)+9.95937277e-01, (float)+9.18494081e-01, (float)+8.06906183e-01, (float)+5.59246168e-01, },
    {(float)+6.32993513e-01, (float)+9.95937277e-01, (float)+9.17717834e-01, (float)+8.06525327e-01, (float)+5.58387289e-01, },
    {(float)+6.32298424e-01, (float)+9.95961036e-01, (float)+9.16747526e-01, (float)+8.06271423e-01, (float)+5.57528409e-01, },
    {(float)+6.31603336e-01, (float)+9.95961036e-01, (float)+9.15971279e-01, (float)+8.06017519e-01, (float)+5.56669530e-01, },
    {(float)+6.30908248e-01, (float)+9.95961036e-01, (float)+9.15195032e-01, (float)+8.05636664e-01, (float)+5.55810650e-01, },
    {(float)+6.30213160e-01, (float)+9.95984794e-01, (float)+9.14418785e-01, (float)+8.05382760e-01, (float)+5.54935254e-01, },
    {(float)+6.29518072e-01, (float)+9.95984794e-01, (float)+9.13642538e-01, (float)+8.05128856e-01, (float)+5.54076374e-01, },
    {(float)+6.28822984e-01, (float)+9.95984794e-01, (float)+9.12672230e-01, (float)+8.04748001e-01, (float)+5.53217495e-01, },
    {(float)+6.28127896e-01, (float)+9.96008553e-01, (float)+9.11895983e-01, (float)+8.04494097e-01, (float)+5.52358615e-01, },
    {(float)+6.27432808e-01, (float)+9.96008553e-01, (float)+9.11119736e-01, (float)+8.04113241e-01, (float)+5.51499736e-01, },
    {(float)+6.26737720e-01, (float)+9.96008553e-01, (float)+9.10343489e-01, (float)+8.03859337e-01, (float)+5.50640856e-01, },
    {(float)+6.26042632e-01, (float)+9.96032312e-01, (float)+9.09567242e-01, (float)+8.03605434e-01, (float)+5.49781977e-01, },
    {(float)+6.25347544e-01, (float)+9.96032312e-01, (float)+9.08790996e-01, (float)+8.03224578e-01, (float)+5.48906580e-01, },
    {(float)+6.24652456e-01, (float)+9.96032312e-01, (float)+9.07820687e-01, (float)+8.02970674e-01, (float)+5.48047701e-01, },
    {(float)+6.23957368e-01, (float)+9.96056070e-01, (float)+9.07044440e-01, (float)+8.02589818e-01, (float)+5.47188821e-01, },
    {(float)+6.23262280e-01, (float)+9.96056070e-01, (float)+9.06268193e-01, (float)+8.02335915e-01, (float)+5.46329942e-01, },
    {(float)+6.22567192e-01, (float)+9.96056070e-01, (float)+9.05491946e-01, (float)+8.02082011e-01, (float)+5.45471062e-01, },
    {(float)+6.21872104e-01, (float)+9.96079829e-01, (float)+9.04715700e-01, (float)+8.01701155e-01, (float)+5.44612183e-01, },
    {(float)+6.21177016e-01, (float)+9.96079829e-01, (float)+9.03745391e-01, (float)+8.01447251e-01, (float)+5.43753303e-01, },
    {(float)+6.20481928e-01, (float)+9.96079829e-01, (float)+9.02969144e-01, (float)+8.01193348e-01, (float)+5.42877907e-01, },
    {(float)+6.19786840e-01, (float)+9.96103588e-01, (float)+9.02192897e-01, (float)+8.00812492e-01, (float)+5.42019027e-01, },
    {(float)+6.18975904e-01, (float)+9.96103588e-01, (float)+9.01416650e-01, (float)+8.00558588e-01, (float)+5.41160148e-01, },
    {(float)+6.18280816e-01, (float)+9.96103588e-01, (float)+9.00640404e-01, (float)+8.00177733e-01, (float)+5.40301268e-01, },
    {(float)+6.17585728e-01, (float)+9.96127346e-01, (float)+8.99670095e-01, (float)+7.99923829e-01, (float)+5.39442389e-01, },
    {(float)+6.16890639e-01, (float)+9.96127346e-01, (float)+8.98893848e-01, (float)+7.99669925e-01, (float)+5.38583510e-01, },
    {(float)+6.16195551e-01, (float)+9.96127346e-01, (float)+8.98117601e-01, (float)+7.99289069e-01, (float)+5.37724630e-01, },
    {(float)+6.15500463e-01, (float)+9.96151105e-01, (float)+8.97341355e-01, (float)+7.99035166e-01, (float)+5.36849234e-01, },
    {(float)+6.14805375e-01, (float)+9.96151105e-01, (float)+8.96565108e-01, (float)+7.98781262e-01, (float)+5.35990354e-01, },
    {(float)+6.14110287e-01, (float)+9.96151105e-01, (float)+8.95594799e-01, (float)+7.98400406e-01, (float)+5.35131475e-01, },
    {(float)+6.13415199e-01, (float)+9.96174863e-01, (float)+8.94818552e-01, (float)+7.98146502e-01, (float)+5.34272595e-01, },
    {(float)+6.12720111e-01, (float)+9.96174863e-01, (float)+8.94042305e-01, (float)+7.97765647e-01, (float)+5.33413716e-01, },
    {(float)+6.12025023e-01, (float)+9.96174863e-01, (float)+8.93266059e-01, (float)+7.97511743e-01, (float)+5.32554836e-01, },
    {(float)+6.11329935e-01, (float)+9.96198622e-01, (float)+8.92489812e-01, (float)+7.97257839e-01, (float)+5.31695957e-01, },
    {(float)+6.10634847e-01, (float)+9.96198622e-01, (float)+8.91519503e-01, (float)+7.96876984e-01, (float)+5.30820560e-01, },
    {(float)+6.09939759e-01, (float)+9.96198622e-01, (float)+8.90743256e-01, (float)+7.96623080e-01, (float)+5.29961681e-01, },
    {(float)+6.09244671e-01, (float)+9.96222381e-01, (float)+8.89967010e-01, (float)+7.96242224e-01, (float)+5.29102801e-01, },
    {(float)+6.08549583e-01, (float)+9.96222381e-01, (float)+8.89190763e-01, (float)+7.95988320e-01, (float)+5.28243922e-01, },
    {(float)+6.07854495e-01, (float)+9.96246139e-01, (float)+8.88414516e-01, (float)+7.95734417e-01, (float)+5.27385042e-01, },
    {(float)+6.07159407e-01, (float)+9.96246139e-01, (float)+8.87444207e-01, (float)+7.95353561e-01, (float)+5.26526163e-01, },
    {(float)+6.06464319e-01, (float)+9.96246139e-01, (float)+8.86667960e-01, (float)+7.95099657e-01, (float)+5.25667283e-01, },
    {(float)+6.05769231e-01, (float)+9.96269898e-01, (float)+8.85891714e-01, (float)+7.94845753e-01, (float)+5.24791887e-01, },
    {(float)+6.05074143e-01, (float)+9.96269898e-01, (float)+8.85115467e-01, (float)+7.94464898e-01, (float)+5.23933007e-01, },
    {(float)+6.04379055e-01, (float)+9.96269898e-01, (float)+8.84339220e-01, (float)+7.94210994e-01, (float)+5.23074128e-01, },
    {(float)+6.03683967e-01, (float)+9.96293656e-01, (float)+8.83368911e-01, (float)+7.93830138e-01, (float)+5.22215248e-01, },
    {(float)+6.02988879e-01, (float)+9.96293656e-01, (float)+8.82592664e-01, (float)+7.93576235e-01, (float)+5.21356369e-01, },
    {(float)+6.02293791e-01, (float)+9.96293656e-01, (float)+8.81816418e-01, (float)+7.93322331e-01, (float)+5.20497489e-01, },
    {(float)+6.01598703e-01, (float)+9.96317415e-01, (float)+8.81040171e-01, (float)+7.92941475e-01, (float)+5.19638610e-01, },
    {(float)+6.00903614e-01, (float)+9.96317415e-01, (float)+8.80263924e-01, (float)+7.92687571e-01, (float)+5.18779730e-01, },
    {(float)+6.00208526e-01, (float)+9.96317415e-01, (float)+8.79293615e-01, (float)+7.92306716e-01, (float)+5.17904334e-01, },
    {(float)+5.99513438e-01, (float)+9.96341174e-01, (float)+8.78517369e-01, (float)+7.92052812e-01, (float)+5.17045455e-01, },
    {(float)+5.98818350e-01, (float)+9.96341174e-01, (float)+8.77741122e-01, (float)+7.91798908e-01, (float)+5.16186575e-01, },
    {(float)+5.98123262e-01, (float)+9.96341174e-01, (float)+8.76964875e-01, (float)+7.91418053e-01, (float)+5.15327696e-01, },
    {(float)+5.97428174e-01, (float)+9.96364932e-01, (float)+8.76188628e-01, (float)+7.91164149e-01, (float)+5.14468816e-01, },
    {(float)+5.96733086e-01, (float)+9.96364932e-01, (float)+8.75218319e-01, (float)+7.90910245e-01, (float)+5.13609937e-01, },
    {(float)+5.96037998e-01, (float)+9.96364932e-01, (float)+8.74442073e-01, (float)+7.90529389e-01, (float)+5.12751057e-01, },
    {(float)+5.95342910e-01, (float)+9.96388691e-01, (float)+8.73665826e-01, (float)+7.90275486e-01, (float)+5.11875661e-01, },
    {(float)+5.94647822e-01, (float)+9.96388691e-01, (float)+8.72889579e-01, (float)+7.89894630e-01, (float)+5.11016781e-01, },
    {(float)+5.93952734e-01, (float)+9.96388691e-01, (float)+8.72113332e-01, (float)+7.89640726e-01, (float)+5.10157902e-01, },
    {(float)+5.93257646e-01, (float)+9.96412450e-01, (float)+8.71143023e-01, (float)+7.89386822e-01, (float)+5.09299022e-01, },
    {(float)+5.92562558e-01, (float)+9.96412450e-01, (float)+8.70366777e-01, (float)+7.89005967e-01, (float)+5.08440143e-01, },
    {(float)+5.91867470e-01, (float)+9.96412450e-01, (float)+8.69590530e-01, (float)+7.88752063e-01, (float)+5.07581263e-01, },
    {(float)+5.91172382e-01, (float)+9.96436208e-01, (float)+8.68814283e-01, (float)+7.88498159e-01, (float)+5.06722384e-01, },
    {(float)+5.90477294e-01, (float)+9.96436208e-01, (float)+8.68038036e-01, (float)+7.88117304e-01, (float)+5.05846987e-01, },
    {(float)+5.89782206e-01, (float)+9.96436208e-01, (float)+8.67067728e-01, (float)+7.87863400e-01, (float)+5.04988108e-01, },
    {(float)+5.89087118e-01, (float)+9.96459967e-01, (float)+8.66291481e-01, (float)+7.87482544e-01, (float)+5.04129228e-01, },
    {(float)+5.88392030e-01, (float)+9.96459967e-01, (float)+8.65515234e-01, (float)+7.87228640e-01, (float)+5.03270349e-01, },
    {(float)+5.87696942e-01, (float)+9.96459967e-01, (float)+8.64738987e-01, (float)+7.86974737e-01, (float)+5.02411469e-01, },
    {(float)+5.87001854e-01, (float)+9.96483725e-01, (float)+8.63962740e-01, (float)+7.86593881e-01, (float)+5.01552590e-01, },
    {(float)+5.86306766e-01, (float)+9.96483725e-01, (float)+8.62992432e-01, (float)+7.86339977e-01, (float)+5.00693710e-01, },
    {(float)+5.85611677e-01, (float)+9.96483725e-01, (float)+8.62216185e-01, (float)+7.85959121e-01, (float)+4.99818314e-01, },
    {(float)+5.84916589e-01, (float)+9.96507484e-01, (float)+8.61439938e-01, (float)+7.85705218e-01, (float)+4.98959434e-01, },
    {(float)+5.84221501e-01, (float)+9.96507484e-01, (float)+8.60663691e-01, (float)+7.85451314e-01, (float)+4.98100555e-01, },
    {(float)+5.83526413e-01, (float)+9.96507484e-01, (float)+8.59887444e-01, (float)+7.85070458e-01, (float)+4.97241675e-01, },
    {(float)+5.82831325e-01, (float)+9.96531243e-01, (float)+8.58917136e-01, (float)+7.84816555e-01, (float)+4.96382796e-01, },
    {(float)+5.82136237e-01, (float)+9.96531243e-01, (float)+8.58140889e-01, (float)+7.84562651e-01, (float)+4.95523916e-01, },
    {(float)+5.81441149e-01, (float)+9.96531243e-01, (float)+8.57364642e-01, (float)+7.84181795e-01, (float)+4.94665037e-01, },
    {(float)+5.80746061e-01, (float)+9.96555001e-01, (float)+8.56588395e-01, (float)+7.83927891e-01, (float)+4.93789641e-01, },
    {(float)+5.80050973e-01, (float)+9.96555001e-01, (float)+8.55812148e-01, (float)+7.83547036e-01, (float)+4.92930761e-01, },
    {(float)+5.79355885e-01, (float)+9.96555001e-01, (float)+8.55035901e-01, (float)+7.83293132e-01, (float)+4.92071882e-01, },
    {(float)+5.78660797e-01, (float)+9.96578760e-01, (float)+8.54065593e-01, (float)+7.83039228e-01, (float)+4.91213002e-01, },
    {(float)+5.77965709e-01, (float)+9.96578760e-01, (float)+8.53289346e-01, (float)+7.82658372e-01, (float)+4.90354123e-01, },
    {(float)+5.77270621e-01, (float)+9.96578760e-01, (float)+8.52513099e-01, (float)+7.82404469e-01, (float)+4.89495243e-01, },
    {(float)+5.76575533e-01, (float)+9.96602518e-01, (float)+8.51736852e-01, (float)+7.82023613e-01, (float)+4.88636364e-01, },
    {(float)+5.75880445e-01, (float)+9.96602518e-01, (float)+8.50960605e-01, (float)+7.81769709e-01, (float)+4.87760967e-01, },
    {(float)+5.75185357e-01, (float)+9.96602518e-01, (float)+8.49990297e-01, (float)+7.81515806e-01, (float)+4.86902088e-01, },
    {(float)+5.74490269e-01, (float)+9.96626277e-01, (float)+8.49214050e-01, (float)+7.81134950e-01, (float)+4.86043208e-01, },
    {(float)+5.73795181e-01, (float)+9.96626277e-01, (float)+8.48437803e-01, (float)+7.80881046e-01, (float)+4.85184329e-01, },
    {(float)+5.73100093e-01, (float)+9.96626277e-01, (float)+8.47661556e-01, (float)+7.80627142e-01, (float)+4.84325449e-01, },
    {(float)+5.72405005e-01, (float)+9.96650036e-01, (float)+8.46885310e-01, (float)+7.80246287e-01, (float)+4.83466570e-01, },
    {(float)+5.71709917e-01, (float)+9.96650036e-01, (float)+8.45915001e-01, (float)+7.79992383e-01, (float)+4.82607690e-01, },
    {(float)+5.71014829e-01, (float)+9.96650036e-01, (float)+8.45138754e-01, (float)+7.79611527e-01, (float)+4.81732294e-01, },
    {(float)+5.70319741e-01, (float)+9.96673794e-01, (float)+8.44362507e-01, (float)+7.79357623e-01, (float)+4.80873414e-01, },
    {(float)+5.69624652e-01, (float)+9.96673794e-01, (float)+8.43586260e-01, (float)+7.79103720e-01, (float)+4.80014535e-01, },
    {(float)+5.68929564e-01, (float)+9.96673794e-01, (float)+8.42810014e-01, (float)+7.78722864e-01, (float)+4.79155655e-01, },
    {(float)+5.68234476e-01, (float)+9.96697553e-01, (float)+8.41839705e-01, (float)+7.78468960e-01, (float)+4.78296776e-01, },
    {(float)+5.67539388e-01, (float)+9.96697553e-01, (float)+8.41063458e-01, (float)+7.78215056e-01, (float)+4.77437896e-01, },
    {(float)+5.66728452e-01, (float)+9.96697553e-01, (float)+8.40287211e-01, (float)+7.77834201e-01, (float)+4.76579017e-01, },
    {(float)+5.66033364e-01, (float)+9.96721311e-01, (float)+8.39510964e-01, (float)+7.77580297e-01, (float)+4.75703621e-01, },
    {(float)+5.65338276e-01, (float)+9.96721311e-01, (float)+8.38734718e-01, (float)+7.77199441e-01, (float)+4.74844741e-01, },
    {(float)+5.64643188e-01, (float)+9.96721311e-01, (float)+8.37764409e-01, (float)+7.76945538e-01, (float)+4.73985862e-01, },
    {(float)+5.63948100e-01, (float)+9.96745070e-01, (float)+8.36988162e-01, (float)+7.76691634e-01, (float)+4.73126982e-01, },
    {(float)+5.63253012e-01, (float)+9.96745070e-01, (float)+8.36211915e-01, (float)+7.76310778e-01, (float)+4.72268103e-01, },
    {(float)+5.62557924e-01, (float)+9.96745070e-01, (float)+8.35435669e-01, (float)+7.76056874e-01, (float)+4.71409223e-01, },
    {(float)+5.61862836e-01, (float)+9.96768829e-01, (float)+8.34659422e-01, (float)+7.75676019e-01, (float)+4.70550344e-01, },
    {(float)+5.61167748e-01, (float)+9.96768829e-01, (float)+8.33689113e-01, (float)+7.75422115e-01, (float)+4.69691464e-01, },
    {(float)+5.60472660e-01, (float)+9.96768829e-01, (float)+8.32912866e-01, (float)+7.75168211e-01, (float)+4.68816068e-01, },
    {(float)+5.59777572e-01, (float)+9.96792587e-01, (float)+8.32136619e-01, (float)+7.74787356e-01, (float)+4.67957188e-01, },
    {(float)+5.59082484e-01, (float)+9.96792587e-01, (float)+8.31360373e-01, (float)+7.74533452e-01, (float)+4.67098309e-01, },
    {(float)+5.58387396e-01, (float)+9.96792587e-01, (float)+8.30584126e-01, (float)+7.74279548e-01, (float)+4.66239429e-01, },
    {(float)+5.57692308e-01, (float)+9.96816346e-01, (float)+8.29613817e-01, (float)+7.73898692e-01, (float)+4.65380550e-01, },
    {(float)+5.56997220e-01, (float)+9.96816346e-01, (float)+8.28837570e-01, (float)+7.73644789e-01, (float)+4.64521670e-01, },
    {(float)+5.56302132e-01, (float)+9.96816346e-01, (float)+8.28061324e-01, (float)+7.73263933e-01, (float)+4.63662791e-01, },
    {(float)+5.55607044e-01, (float)+9.96840105e-01, (float)+8.27285077e-01, (float)+7.73010029e-01, (float)+4.62787394e-01, },
    {(float)+5.54911956e-01, (float)+9.96840105e-01, (float)+8.26508830e-01, (float)+7.72756125e-01, (float)+4.61928515e-01, },
    {(float)+5.54216867e-01, (float)+9.96840105e-01, (float)+8.25538521e-01, (float)+7.72375270e-01, (float)+4.61069635e-01, },
    {(float)+5.53521779e-01, (float)+9.96863863e-01, (float)+8.24762274e-01, (float)+7.72121366e-01, (float)+4.60210756e-01, },
    {(float)+5.52826691e-01, (float)+9.96863863e-01, (float)+8.23986028e-01, (float)+7.71740510e-01, (float)+4.59351876e-01, },
    {(float)+5.52131603e-01, (float)+9.96863863e-01, (float)+8.23209781e-01, (float)+7.71486607e-01, (float)+4.58492997e-01, },
    {(float)+5.51436515e-01, (float)+9.96887622e-01, (float)+8.22433534e-01, (float)+7.71232703e-01, (float)+4.57634117e-01, },
    {(float)+5.50741427e-01, (float)+9.96887622e-01, (float)+8.21463225e-01, (float)+7.70851847e-01, (float)+4.56758721e-01, },
    {(float)+5.50046339e-01, (float)+9.96887622e-01, (float)+8.20686978e-01, (float)+7.70597943e-01, (float)+4.55899841e-01, },
    {(float)+5.49351251e-01, (float)+9.96911380e-01, (float)+8.19910732e-01, (float)+7.70344040e-01, (float)+4.55040962e-01, },
    {(float)+5.48656163e-01, (float)+9.96911380e-01, (float)+8.19134485e-01, (float)+7.69963184e-01, (float)+4.54182082e-01, },
    {(float)+5.47961075e-01, (float)+9.96935139e-01, (float)+8.18358238e-01, (float)+7.69709280e-01, (float)+4.53323203e-01, },
    {(float)+5.47265987e-01, (float)+9.96935139e-01, (float)+8.17387929e-01, (float)+7.69328425e-01, (float)+4.52464323e-01, },
    {(float)+5.46570899e-01, (float)+9.96935139e-01, (float)+8.16611683e-01, (float)+7.69074521e-01, (float)+4.51605444e-01, },
    {(float)+5.45875811e-01, (float)+9.96958898e-01, (float)+8.15835436e-01, (float)+7.68820617e-01, (float)+4.50730048e-01, },
    {(float)+5.45180723e-01, (float)+9.96958898e-01, (float)+8.15059189e-01, (float)+7.68439761e-01, (float)+4.49871168e-01, },
    {(float)+5.44485635e-01, (float)+9.96958898e-01, (float)+8.14282942e-01, (float)+7.68185858e-01, (float)+4.49012289e-01, },
    {(float)+5.43790547e-01, (float)+9.96982656e-01, (float)+8.13312633e-01, (float)+7.67931954e-01, (float)+4.48153409e-01, },
    {(float)+5.43095459e-01, (float)+9.96982656e-01, (float)+8.12536387e-01, (float)+7.67551098e-01, (float)+4.47294530e-01, },
    {(float)+5.42400371e-01, (float)+9.96982656e-01, (float)+8.11760140e-01, (float)+7.67297194e-01, (float)+4.46435650e-01, },
    {(float)+5.41705283e-01, (float)+9.97006415e-01, (float)+8.10983893e-01, (float)+7.66916339e-01, (float)+4.45576771e-01, },
    {(float)+5.41010195e-01, (float)+9.97006415e-01, (float)+8.10207646e-01, (float)+7.66662435e-01, (float)+4.44701374e-01, },
    {(float)+5.40315107e-01, (float)+9.97006415e-01, (float)+8.09237337e-01, (float)+7.66408531e-01, (float)+4.43842495e-01, },
    {(float)+5.39620019e-01, (float)+9.97030173e-01, (float)+8.08461091e-01, (float)+7.66027676e-01, (float)+4.42983615e-01, },
    {(float)+5.38924930e-01, (float)+9.97030173e-01, (float)+8.07684844e-01, (float)+7.65773772e-01, (float)+4.42124736e-01, },
    {(float)+5.38229842e-01, (float)+9.97030173e-01, (float)+8.06908597e-01, (float)+7.65392916e-01, (float)+4.41265856e-01, },
    {(float)+5.37534754e-01, (float)+9.97053932e-01, (float)+8.06132350e-01, (float)+7.65139012e-01, (float)+4.40406977e-01, },
    {(float)+5.36839666e-01, (float)+9.97053932e-01, (float)+8.05162042e-01, (float)+7.64885109e-01, (float)+4.39548097e-01, },
    {(float)+5.36144578e-01, (float)+9.97053932e-01, (float)+8.04385795e-01, (float)+7.64504253e-01, (float)+4.38672701e-01, },
    {(float)+5.35449490e-01, (float)+9.97077691e-01, (float)+8.03609548e-01, (float)+7.64250349e-01, (float)+4.37813821e-01, },
    {(float)+5.34754402e-01, (float)+9.97077691e-01, (float)+8.02833301e-01, (float)+7.63996445e-01, (float)+4.36954942e-01, },
    {(float)+5.34059314e-01, (float)+9.97077691e-01, (float)+8.02057054e-01, (float)+7.63615590e-01, (float)+4.36096062e-01, },
    {(float)+5.33364226e-01, (float)+9.97101449e-01, (float)+8.01280807e-01, (float)+7.63361686e-01, (float)+4.35237183e-01, },
    {(float)+5.32669138e-01, (float)+9.97101449e-01, (float)+8.00310499e-01, (float)+7.62980830e-01, (float)+4.34378303e-01, },
    {(float)+5.31974050e-01, (float)+9.97101449e-01, (float)+7.99534252e-01, (float)+7.62726926e-01, (float)+4.33519424e-01, },
    {(float)+5.31278962e-01, (float)+9.97125208e-01, (float)+7.98758005e-01, (float)+7.62473023e-01, (float)+4.32644027e-01, },
    {(float)+5.30583874e-01, (float)+9.97125208e-01, (float)+7.97981758e-01, (float)+7.62092167e-01, (float)+4.31785148e-01, },
    {(float)+5.29888786e-01, (float)+9.97125208e-01, (float)+7.97205511e-01, (float)+7.61838263e-01, (float)+4.30926268e-01, },
    {(float)+5.29193698e-01, (float)+9.97148967e-01, (float)+7.96235203e-01, (float)+7.61457408e-01, (float)+4.30067389e-01, },
    {(float)+5.28498610e-01, (float)+9.97148967e-01, (float)+7.95458956e-01, (float)+7.61203504e-01, (float)+4.29208510e-01, },
    {(float)+5.27803522e-01, (float)+9.97148967e-01, (float)+7.94682709e-01, (float)+7.60949600e-01, (float)+4.28349630e-01, },
    {(float)+5.27108434e-01, (float)+9.97172725e-01, (float)+7.93906462e-01, (float)+7.60568744e-01, (float)+4.27490751e-01, },
    {(float)+5.26413346e-01, (float)+9.97172725e-01, (float)+7.93130215e-01, (float)+7.60314841e-01, (float)+4.26631871e-01, },
    {(float)+5.25718258e-01, (float)+9.97172725e-01, (float)+7.92159907e-01, (float)+7.60060937e-01, (float)+4.25756475e-01, },
    {(float)+5.25023170e-01, (float)+9.97196484e-01, (float)+7.91383660e-01, (float)+7.59680081e-01, (float)+4.24897595e-01, },
    {(float)+5.24328082e-01, (float)+9.97196484e-01, (float)+7.90607413e-01, (float)+7.59426177e-01, (float)+4.24038716e-01, },
    {(float)+5.23632994e-01, (float)+9.97196484e-01, (float)+7.89831166e-01, (float)+7.59045322e-01, (float)+4.23179836e-01, },
    {(float)+5.22937905e-01, (float)+9.97220242e-01, (float)+7.89054919e-01, (float)+7.58791418e-01, (float)+4.22320957e-01, },
    {(float)+5.22242817e-01, (float)+9.97220242e-01, (float)+7.88084611e-01, (float)+7.58537514e-01, (float)+4.21462077e-01, },
    {(float)+5.21547729e-01, (float)+9.97220242e-01, (float)+7.87308364e-01, (float)+7.58156659e-01, (float)+4.20603198e-01, },
    {(float)+5.20852641e-01, (float)+9.97244001e-01, (float)+7.86532117e-01, (float)+7.57902755e-01, (float)+4.19727801e-01, },
    {(float)+5.20157553e-01, (float)+9.97244001e-01, (float)+7.85755870e-01, (float)+7.57648851e-01, (float)+4.18868922e-01, },
    {(float)+5.19462465e-01, (float)+9.97244001e-01, (float)+7.84979624e-01, (float)+7.57267995e-01, (float)+4.18010042e-01, },
    {(float)+5.18767377e-01, (float)+9.97267760e-01, (float)+7.84009315e-01, (float)+7.57014092e-01, (float)+4.17151163e-01, },
    {(float)+5.18072289e-01, (float)+9.97267760e-01, (float)+7.83233068e-01, (float)+7.56633236e-01, (float)+4.16292283e-01, },
    {(float)+5.17377201e-01, (float)+9.97267760e-01, (float)+7.82456821e-01, (float)+7.56379332e-01, (float)+4.15433404e-01, },
    {(float)+5.16682113e-01, (float)+9.97291518e-01, (float)+7.81680574e-01, (float)+7.56125428e-01, (float)+4.14574524e-01, },
    {(float)+5.15987025e-01, (float)+9.97291518e-01, (float)+7.80904328e-01, (float)+7.55744573e-01, (float)+4.13699128e-01, },
    {(float)+5.15291937e-01, (float)+9.97291518e-01, (float)+7.79934019e-01, (float)+7.55490669e-01, (float)+4.12840248e-01, },
    {(float)+5.14481001e-01, (float)+9.97315277e-01, (float)+7.79157772e-01, (float)+7.55109813e-01, (float)+4.11981369e-01, },
    {(float)+5.13785913e-01, (float)+9.97315277e-01, (float)+7.78381525e-01, (float)+7.54855910e-01, (float)+4.11122489e-01, },
    {(float)+5.13090825e-01, (float)+9.97315277e-01, (float)+7.77605278e-01, (float)+7.54602006e-01, (float)+4.10263610e-01, },
    {(float)+5.12395737e-01, (float)+9.97339035e-01, (float)+7.76829032e-01, (float)+7.54221150e-01, (float)+4.09404730e-01, },
    {(float)+5.11700649e-01, (float)+9.97339035e-01, (float)+7.75858723e-01, (float)+7.53967246e-01, (float)+4.08545851e-01, },
    {(float)+5.11005561e-01, (float)+9.97339035e-01, (float)+7.75082476e-01, (float)+7.53713343e-01, (float)+4.07670455e-01, },
    {(float)+5.10310473e-01, (float)+9.97362794e-01, (float)+7.74306229e-01, (float)+7.53332487e-01, (float)+4.06811575e-01, },
    {(float)+5.09615385e-01, (float)+9.97362794e-01, (float)+7.73529983e-01, (float)+7.53078583e-01, (float)+4.05952696e-01, },
    {(float)+5.08920297e-01, (float)+9.97362794e-01, (float)+7.72753736e-01, (float)+7.52697728e-01, (float)+4.05093816e-01, },
    {(float)+5.08225209e-01, (float)+9.97386553e-01, (float)+7.71783427e-01, (float)+7.52443824e-01, (float)+4.04234937e-01, },
    {(float)+5.07530120e-01, (float)+9.97386553e-01, (float)+7.71007180e-01, (float)+7.52189920e-01, (float)+4.03376057e-01, },
    {(float)+5.06835032e-01, (float)+9.97386553e-01, (float)+7.70230933e-01, (float)+7.51809064e-01, (float)+4.02517178e-01, },
    {(float)+5.06139944e-01, (float)+9.97410311e-01, (float)+7.69454687e-01, (float)+7.51555161e-01, (float)+4.01641781e-01, },
    {(float)+5.05444856e-01, (float)+9.97410311e-01, (float)+7.68678440e-01, (float)+7.51174305e-01, (float)+4.00782902e-01, },
    {(float)+5.04749768e-01, (float)+9.97410311e-01, (float)+7.67708131e-01, (float)+7.50920401e-01, (float)+3.99924022e-01, },
    {(float)+5.04054680e-01, (float)+9.97434070e-01, (float)+7.66931884e-01, (float)+7.50666497e-01, (float)+3.99065143e-01, },
    {(float)+5.03359592e-01, (float)+9.97434070e-01, (float)+7.66155637e-01, (float)+7.50285642e-01, (float)+3.98206263e-01, },
    {(float)+5.02664504e-01, (float)+9.97434070e-01, (float)+7.65379391e-01, (float)+7.50031738e-01, (float)+3.97347384e-01, },
    {(float)+5.01969416e-01, (float)+9.97457828e-01, (float)+7.64603144e-01, (float)+7.49777834e-01, (float)+3.96488504e-01, },
    {(float)+5.01274328e-01, (float)+9.97457828e-01, (float)+7.63632835e-01, (float)+7.49396979e-01, (float)+3.95613108e-01, },
    {(float)+5.00579240e-01, (float)+9.97457828e-01, (float)+7.62856588e-01, (float)+7.49143075e-01, (float)+3.94754228e-01, },
    {(float)+4.99884152e-01, (float)+9.97481587e-01, (float)+7.62080342e-01, (float)+7.48762219e-01, (float)+3.93895349e-01, },
    {(float)+4.99189064e-01, (float)+9.97481587e-01, (float)+7.61304095e-01, (float)+7.48508315e-01, (float)+3.93036469e-01, },
    {(float)+4.98493976e-01, (float)+9.97481587e-01, (float)+7.60527848e-01, (float)+7.48254412e-01, (float)+3.92177590e-01, },
    {(float)+4.97798888e-01, (float)+9.97505346e-01, (float)+7.59557539e-01, (float)+7.47873556e-01, (float)+3.91318710e-01, },
    {(float)+4.97103800e-01, (float)+9.97505346e-01, (float)+7.58781292e-01, (float)+7.47619652e-01, (float)+3.90459831e-01, },
    {(float)+4.96408712e-01, (float)+9.97505346e-01, (float)+7.58005046e-01, (float)+7.47365748e-01, (float)+3.89584434e-01, },
    {(float)+4.95713624e-01, (float)+9.97529104e-01, (float)+7.57228799e-01, (float)+7.46984893e-01, (float)+3.88725555e-01, },
    {(float)+4.95018536e-01, (float)+9.97529104e-01, (float)+7.56452552e-01, (float)+7.46730989e-01, (float)+3.87866675e-01, },
    {(float)+4.94323448e-01, (float)+9.97529104e-01, (float)+7.55482243e-01, (float)+7.46350133e-01, (float)+3.87007796e-01, },
    {(float)+4.93628360e-01, (float)+9.97552863e-01, (float)+7.54705997e-01, (float)+7.46096230e-01, (float)+3.86148916e-01, },
    {(float)+4.92933272e-01, (float)+9.97552863e-01, (float)+7.53929750e-01, (float)+7.45842326e-01, (float)+3.85290037e-01, },
    {(float)+4.92238184e-01, (float)+9.97552863e-01, (float)+7.53153503e-01, (float)+7.45461470e-01, (float)+3.84431158e-01, },
    {(float)+4.91543095e-01, (float)+9.97576622e-01, (float)+7.52377256e-01, (float)+7.45207566e-01, (float)+3.83555761e-01, },
    {(float)+4.90848007e-01, (float)+9.97576622e-01, (float)+7.51406947e-01, (float)+7.44826711e-01, (float)+3.82696882e-01, },
    {(float)+4.90152919e-01, (float)+9.97576622e-01, (float)+7.50630701e-01, (float)+7.44572807e-01, (float)+3.81838002e-01, },
    {(float)+4.89457831e-01, (float)+9.97600380e-01, (float)+7.49854454e-01, (float)+7.44318903e-01, (float)+3.80979123e-01, },
    {(float)+4.88762743e-01, (float)+9.97600380e-01, (float)+7.49078207e-01, (float)+7.43938047e-01, (float)+3.80120243e-01, },
    {(float)+4.88067655e-01, (float)+9.97624139e-01, (float)+7.48301960e-01, (float)+7.43684144e-01, (float)+3.79261364e-01, },
    {(float)+4.87372567e-01, (float)+9.97624139e-01, (float)+7.47525713e-01, (float)+7.43430240e-01, (float)+3.78402484e-01, },
    {(float)+4.86677479e-01, (float)+9.97624139e-01, (float)+7.46555405e-01, (float)+7.43049384e-01, (float)+3.77543605e-01, },
    {(float)+4.85982391e-01, (float)+9.97647897e-01, (float)+7.45779158e-01, (float)+7.42795481e-01, (float)+3.76668208e-01, },
    {(float)+4.85287303e-01, (float)+9.97647897e-01, (float)+7.45002911e-01, (float)+7.42414625e-01, (float)+3.75809329e-01, },
    {(float)+4.84592215e-01, (float)+9.97647897e-01, (float)+7.44226664e-01, (float)+7.42160721e-01, (float)+3.74950449e-01, },
    {(float)+4.83897127e-01, (float)+9.97671656e-01, (float)+7.43450417e-01, (float)+7.41906817e-01, (float)+3.74091570e-01, },
    {(float)+4.83202039e-01, (float)+9.97671656e-01, (float)+7.42480109e-01, (float)+7.41525962e-01, (float)+3.73232690e-01, },
    {(float)+4.82506951e-01, (float)+9.97671656e-01, (float)+7.41703862e-01, (float)+7.41272058e-01, (float)+3.72373811e-01, },
    {(float)+4.81811863e-01, (float)+9.97695415e-01, (float)+7.40927615e-01, (float)+7.40891202e-01, (float)+3.71514931e-01, },
    {(float)+4.81116775e-01, (float)+9.97695415e-01, (float)+7.40151368e-01, (float)+7.40637298e-01, (float)+3.70639535e-01, },
    {(float)+4.80421687e-01, (float)+9.97695415e-01, (float)+7.39375121e-01, (float)+7.40383395e-01, (float)+3.69780655e-01, },
    {(float)+4.79726599e-01, (float)+9.97719173e-01, (float)+7.38404813e-01, (float)+7.40002539e-01, (float)+3.68921776e-01, },
    {(float)+4.79031511e-01, (float)+9.97719173e-01, (float)+7.37628566e-01, (float)+7.39748635e-01, (float)+3.68062896e-01, },
    {(float)+4.78336423e-01, (float)+9.97719173e-01, (float)+7.36852319e-01, (float)+7.39494731e-01, (float)+3.67204017e-01, },
    {(float)+4.77641335e-01, (float)+9.97742932e-01, (float)+7.36076072e-01, (float)+7.39113876e-01, (float)+3.66345137e-01, },
    {(float)+4.76946247e-01, (float)+9.97742932e-01, (float)+7.35299825e-01, (float)+7.38859972e-01, (float)+3.65486258e-01, },
    {(float)+4.76251158e-01, (float)+9.97742932e-01, (float)+7.34329517e-01, (float)+7.38479116e-01, (float)+3.64610862e-01, },
    {(float)+4.75556070e-01, (float)+9.97766690e-01, (float)+7.33553270e-01, (float)+7.38225213e-01, (float)+3.63751982e-01, },
    {(float)+4.74860982e-01, (float)+9.97766690e-01, (float)+7.32777023e-01, (float)+7.37971309e-01, (float)+3.62893103e-01, },
    {(float)+4.74165894e-01, (float)+9.97766690e-01, (float)+7.32000776e-01, (float)+7.37590453e-01, (float)+3.62034223e-01, },
    {(float)+4.73470806e-01, (float)+9.97790449e-01, (float)+7.31224529e-01, (float)+7.37336549e-01, (float)+3.61175344e-01, },
    {(float)+4.72775718e-01, (float)+9.97790449e-01, (float)+7.30254221e-01, (float)+7.37082646e-01, (float)+3.60316464e-01, },
    {(float)+4.72080630e-01, (float)+9.97790449e-01, (float)+7.29477974e-01, (float)+7.36701790e-01, (float)+3.59457585e-01, },
    {(float)+4.71385542e-01, (float)+9.97814208e-01, (float)+7.28701727e-01, (float)+7.36447886e-01, (float)+3.58582188e-01, },
    {(float)+4.70690454e-01, (float)+9.97814208e-01, (float)+7.27925480e-01, (float)+7.36067031e-01, (float)+3.57723309e-01, },
    {(float)+4.69995366e-01, (float)+9.97814208e-01, (float)+7.27149233e-01, (float)+7.35813127e-01, (float)+3.56864429e-01, },
    {(float)+4.69300278e-01, (float)+9.97837966e-01, (float)+7.26178925e-01, (float)+7.35559223e-01, (float)+3.56005550e-01, },
    {(float)+4.68605190e-01, (float)+9.97837966e-01, (float)+7.25402678e-01, (float)+7.35178367e-01, (float)+3.55146670e-01, },
    {(float)+4.67910102e-01, (float)+9.97837966e-01, (float)+7.24626431e-01, (float)+7.34924464e-01, (float)+3.54287791e-01, },
    {(float)+4.67215014e-01, (float)+9.97861725e-01, (float)+7.23850184e-01, (float)+7.34543608e-01, (float)+3.53428911e-01, },
    {(float)+4.66519926e-01, (float)+9.97861725e-01, (float)+7.23073938e-01, (float)+7.34289704e-01, (float)+3.52553515e-01, },
    {(float)+4.65824838e-01, (float)+9.97861725e-01, (float)+7.22103629e-01, (float)+7.34035800e-01, (float)+3.51694635e-01, },
    {(float)+4.65129750e-01, (float)+9.97885483e-01, (float)+7.21327382e-01, (float)+7.33654945e-01, (float)+3.50835756e-01, },
    {(float)+4.64434662e-01, (float)+9.97885483e-01, (float)+7.20551135e-01, (float)+7.33401041e-01, (float)+3.49976876e-01, },
    {(float)+4.63739574e-01, (float)+9.97885483e-01, (float)+7.19774888e-01, (float)+7.33147137e-01, (float)+3.49117997e-01, },
    {(float)+4.62928638e-01, (float)+9.97909242e-01, (float)+7.18998642e-01, (float)+7.32766282e-01, (float)+3.48259117e-01, },
    {(float)+4.62233550e-01, (float)+9.97909242e-01, (float)+7.18028333e-01, (float)+7.32512378e-01, (float)+3.47400238e-01, },
    {(float)+4.61538462e-01, (float)+9.97909242e-01, (float)+7.17252086e-01, (float)+7.32131522e-01, (float)+3.46524841e-01, },
    {(float)+4.60843373e-01, (float)+9.97933001e-01, (float)+7.16475839e-01, (float)+7.31877618e-01, (float)+3.45665962e-01, },
    {(float)+4.60148285e-01, (float)+9.97933001e-01, (float)+7.15699592e-01, (float)+7.31623715e-01, (float)+3.44807082e-01, },
    {(float)+4.59453197e-01, (float)+9.97933001e-01, (float)+7.14923346e-01, (float)+7.31242859e-01, (float)+3.43948203e-01, },
    {(float)+4.58758109e-01, (float)+9.97956759e-01, (float)+7.13953037e-01, (float)+7.30988955e-01, (float)+3.43089323e-01, },
    {(float)+4.58063021e-01, (float)+9.97956759e-01, (float)+7.13176790e-01, (float)+7.30608100e-01, (float)+3.42230444e-01, },
    {(float)+4.57367933e-01, (float)+9.97956759e-01, (float)+7.12400543e-01, (float)+7.30354196e-01, (float)+3.41371564e-01, },
    {(float)+4.56672845e-01, (float)+9.97980518e-01, (float)+7.11624297e-01, (float)+7.30100292e-01, (float)+3.40496168e-01, },
    {(float)+4.55977757e-01, (float)+9.97980518e-01, (float)+7.10848050e-01, (float)+7.29719436e-01, (float)+3.39637289e-01, },
    {(float)+4.55282669e-01, (float)+9.97980518e-01, (float)+7.09877741e-01, (float)+7.29465533e-01, (float)+3.38778409e-01, },
    {(float)+4.54587581e-01, (float)+9.98004277e-01, (float)+7.09101494e-01, (float)+7.29211629e-01, (float)+3.37919530e-01, },
    {(float)+4.53892493e-01, (float)+9.98004277e-01, (float)+7.08325247e-01, (float)+7.28830773e-01, (float)+3.37060650e-01, },
    {(float)+4.53197405e-01, (float)+9.98004277e-01, (float)+7.07549001e-01, (float)+7.28576869e-01, (float)+3.36201771e-01, },
    {(float)+4.52502317e-01, (float)+9.98028035e-01, (float)+7.06772754e-01, (float)+7.28196014e-01, (float)+3.35342891e-01, },
    {(float)+4.51807229e-01, (float)+9.98028035e-01, (float)+7.05802445e-01, (float)+7.27942110e-01, (float)+3.34467495e-01, },
    {(float)+4.51112141e-01, (float)+9.98028035e-01, (float)+7.05026198e-01, (float)+7.27688206e-01, (float)+3.33608615e-01, },
    {(float)+4.50417053e-01, (float)+9.98051794e-01, (float)+7.04249951e-01, (float)+7.27307351e-01, (float)+3.32749736e-01, },
    {(float)+4.49721965e-01, (float)+9.98051794e-01, (float)+7.03473705e-01, (float)+7.27053447e-01, (float)+3.31890856e-01, },
    {(float)+4.49026877e-01, (float)+9.98051794e-01, (float)+7.02697458e-01, (float)+7.26799543e-01, (float)+3.31031977e-01, },
    {(float)+4.48331789e-01, (float)+9.98075552e-01, (float)+7.01727149e-01, (float)+7.26418687e-01, (float)+3.30173097e-01, },
    {(float)+4.47636701e-01, (float)+9.98075552e-01, (float)+7.00950902e-01, (float)+7.26164784e-01, (float)+3.29314218e-01, },
    {(float)+4.46941613e-01, (float)+9.98075552e-01, (float)+7.00174656e-01, (float)+7.25783928e-01, (float)+3.28455338e-01, },
    {(float)+4.46246525e-01, (float)+9.98099311e-01, (float)+6.99398409e-01, (float)+7.25530024e-01, (float)+3.27579942e-01, },
    {(float)+4.45551437e-01, (float)+9.98099311e-01, (float)+6.98622162e-01, (float)+7.25276120e-01, (float)+3.26721062e-01, },
    {(float)+4.44856348e-01, (float)+9.98099311e-01, (float)+6.97651853e-01, (float)+7.24895265e-01, (float)+3.25862183e-01, },
    {(float)+4.44161260e-01, (float)+9.98123070e-01, (float)+6.96875606e-01, (float)+7.24641361e-01, (float)+3.25003303e-01, },
    {(float)+4.43466172e-01, (float)+9.98123070e-01, (float)+6.96099360e-01, (float)+7.24260505e-01, (float)+3.24144424e-01, },
    {(float)+4.42771084e-01, (float)+9.98123070e-01, (float)+6.95323113e-01, (float)+7.24006601e-01, (float)+3.23285544e-01, },
    {(float)+4.42075996e-01, (float)+9.98146828e-01, (float)+6.94546866e-01, (float)+7.23752698e-01, (float)+3.22426665e-01, },
    {(float)+4.41380908e-01, (float)+9.98146828e-01, (float)+6.93770619e-01, (float)+7.23371842e-01, (float)+3.21551268e-01, },
    {(float)+4.40685820e-01, (float)+9.98146828e-01, (float)+6.92800310e-01, (float)+7.23117938e-01, (float)+3.20692389e-01, },
    {(float)+4.39990732e-01, (float)+9.98170587e-01, (float)+6.92024064e-01, (float)+7.22864035e-01, (float)+3.19833510e-01, },
    {(float)+4.39295644e-01, (float)+9.98170587e-01, (float)+6.91247817e-01, (float)+7.22483179e-01, (float)+3.18974630e-01, },
    {(float)+4.38600556e-01, (float)+9.98170587e-01, (float)+6.90471570e-01, (float)+7.22229275e-01, (float)+3.18115751e-01, },
    {(float)+4.37905468e-01, (float)+9.98194345e-01, (float)+6.89695323e-01, (float)+7.21848419e-01, (float)+3.17256871e-01, },
    {(float)+4.37210380e-01, (float)+9.98194345e-01, (float)+6.88725015e-01, (float)+7.21594516e-01, (float)+3.16397992e-01, },
    {(float)+4.36515292e-01, (float)+9.98194345e-01, (float)+6.87948768e-01, (float)+7.21340612e-01, (float)+3.15522595e-01, },
    {(float)+4.35820204e-01, (float)+9.98218104e-01, (float)+6.87172521e-01, (float)+7.20959756e-01, (float)+3.14663716e-01, },
    {(float)+4.35125116e-01, (float)+9.98218104e-01, (float)+6.86396274e-01, (float)+7.20705852e-01, (float)+3.13804836e-01, },
    {(float)+4.34430028e-01, (float)+9.98218104e-01, (float)+6.85620027e-01, (float)+7.20324997e-01, (float)+3.12945957e-01, },
    {(float)+4.33734940e-01, (float)+9.98241863e-01, (float)+6.84649719e-01, (float)+7.20071093e-01, (float)+3.12087077e-01, },
    {(float)+4.33039852e-01, (float)+9.98241863e-01, (float)+6.83873472e-01, (float)+7.19817189e-01, (float)+3.11228198e-01, },
    {(float)+4.32344764e-01, (float)+9.98241863e-01, (float)+6.83097225e-01, (float)+7.19436334e-01, (float)+3.10369318e-01, },
    {(float)+4.31649676e-01, (float)+9.98265621e-01, (float)+6.82320978e-01, (float)+7.19182430e-01, (float)+3.09493922e-01, },
    {(float)+4.30954588e-01, (float)+9.98265621e-01, (float)+6.81544731e-01, (float)+7.18928526e-01, (float)+3.08635042e-01, },
    {(float)+4.30259500e-01, (float)+9.98265621e-01, (float)+6.80574423e-01, (float)+7.18547670e-01, (float)+3.07776163e-01, },
    {(float)+4.29564411e-01, (float)+9.98289380e-01, (float)+6.79798176e-01, (float)+7.18293767e-01, (float)+3.06917283e-01, },
    {(float)+4.28869323e-01, (float)+9.98289380e-01, (float)+6.79021929e-01, (float)+7.17912911e-01, (float)+3.06058404e-01, },
    {(float)+4.28174235e-01, (float)+9.98313139e-01, (float)+6.78245682e-01, (float)+7.17659007e-01, (float)+3.05199524e-01, },
    {(float)+4.27479147e-01, (float)+9.98313139e-01, (float)+6.77469435e-01, (float)+7.17405103e-01, (float)+3.04340645e-01, },
    {(float)+4.26784059e-01, (float)+9.98313139e-01, (float)+6.76499127e-01, (float)+7.17024248e-01, (float)+3.03465248e-01, },
    {(float)+4.26088971e-01, (float)+9.98336897e-01, (float)+6.75722880e-01, (float)+7.16770344e-01, (float)+3.02606369e-01, },
    {(float)+4.25393883e-01, (float)+9.98336897e-01, (float)+6.74946633e-01, (float)+7.16516440e-01, (float)+3.01747489e-01, },
    {(float)+4.24698795e-01, (float)+9.98336897e-01, (float)+6.74170386e-01, (float)+7.16135585e-01, (float)+3.00888610e-01, },
    {(float)+4.24003707e-01, (float)+9.98360656e-01, (float)+6.73394139e-01, (float)+7.15881681e-01, (float)+3.00029730e-01, },
    {(float)+4.23308619e-01, (float)+9.98360656e-01, (float)+6.72423831e-01, (float)+7.15500825e-01, (float)+2.99170851e-01, },
    {(float)+4.22613531e-01, (float)+9.98360656e-01, (float)+6.71647584e-01, (float)+7.15246921e-01, (float)+2.98311971e-01, },
    {(float)+4.21918443e-01, (float)+9.98384414e-01, (float)+6.70871337e-01, (float)+7.14993018e-01, (float)+2.97436575e-01, },
    {(float)+4.21223355e-01, (float)+9.98384414e-01, (float)+6.70095090e-01, (float)+7.14612162e-01, (float)+2.96577696e-01, },
    {(float)+4.20528267e-01, (float)+9.98384414e-01, (float)+6.69318843e-01, (float)+7.14358258e-01, (float)+2.95718816e-01, },
    {(float)+4.19833179e-01, (float)+9.98408173e-01, (float)+6.68348535e-01, (float)+7.13977403e-01, (float)+2.94859937e-01, },
    {(float)+4.19138091e-01, (float)+9.98408173e-01, (float)+6.67572288e-01, (float)+7.13723499e-01, (float)+2.94001057e-01, },
    {(float)+4.18443003e-01, (float)+9.98408173e-01, (float)+6.66796041e-01, (float)+7.13469595e-01, (float)+2.93142178e-01, },
    {(float)+4.17747915e-01, (float)+9.98431932e-01, (float)+6.66019794e-01, (float)+7.13088739e-01, (float)+2.92283298e-01, },
    {(float)+4.17052827e-01, (float)+9.98431932e-01, (float)+6.65243547e-01, (float)+7.12834836e-01, (float)+2.91407902e-01, },
    {(float)+4.16357739e-01, (float)+9.98431932e-01, (float)+6.64273239e-01, (float)+7.12580932e-01, (float)+2.90549022e-01, },
    {(float)+4.15662651e-01, (float)+9.98455690e-01, (float)+6.63496992e-01, (float)+7.12200076e-01, (float)+2.89690143e-01, },
    {(float)+4.14967563e-01, (float)+9.98455690e-01, (float)+6.62720745e-01, (float)+7.11946172e-01, (float)+2.88831263e-01, },
    {(float)+4.14272475e-01, (float)+9.98455690e-01, (float)+6.61944498e-01, (float)+7.11565317e-01, (float)+2.87972384e-01, },
    {(float)+4.13577386e-01, (float)+9.98479449e-01, (float)+6.61168252e-01, (float)+7.11311413e-01, (float)+2.87113504e-01, },
    {(float)+4.12882298e-01, (float)+9.98479449e-01, (float)+6.60197943e-01, (float)+7.11057509e-01, (float)+2.86254625e-01, },
    {(float)+4.12187210e-01, (float)+9.98479449e-01, (float)+6.59421696e-01, (float)+7.10676654e-01, (float)+2.85395745e-01, },
    {(float)+4.11492122e-01, (float)+9.98503207e-01, (float)+6.58645449e-01, (float)+7.10422750e-01, (float)+2.84520349e-01, },
    {(float)+4.10681186e-01, (float)+9.98503207e-01, (float)+6.57869202e-01, (float)+7.10041894e-01, (float)+2.83661469e-01, },
    {(float)+4.09986098e-01, (float)+9.98503207e-01, (float)+6.57092956e-01, (float)+7.09787990e-01, (float)+2.82802590e-01, },
    {(float)+4.09291010e-01, (float)+9.98526966e-01, (float)+6.56122647e-01, (float)+7.09534087e-01, (float)+2.81943710e-01, },
    {(float)+4.08595922e-01, (float)+9.98526966e-01, (float)+6.55346400e-01, (float)+7.09153231e-01, (float)+2.81084831e-01, },
    {(float)+4.07900834e-01, (float)+9.98526966e-01, (float)+6.54570153e-01, (float)+7.08899327e-01, (float)+2.80225951e-01, },
    {(float)+4.07205746e-01, (float)+9.98550725e-01, (float)+6.53793906e-01, (float)+7.08645423e-01, (float)+2.79367072e-01, },
    {(float)+4.06510658e-01, (float)+9.98550725e-01, (float)+6.53017660e-01, (float)+7.08264568e-01, (float)+2.78491675e-01, },
    {(float)+4.05815570e-01, (float)+9.98550725e-01, (float)+6.52047351e-01, (float)+7.08010664e-01, (float)+2.77632796e-01, },
    {(float)+4.05120482e-01, (float)+9.98574483e-01, (float)+6.51271104e-01, (float)+7.07629808e-01, (float)+2.76773916e-01, },
    {(float)+4.04425394e-01, (float)+9.98574483e-01, (float)+6.50494857e-01, (float)+7.07375905e-01, (float)+2.75915037e-01, },
    {(float)+4.03730306e-01, (float)+9.98574483e-01, (float)+6.49718611e-01, (float)+7.07122001e-01, (float)+2.75056158e-01, },
    {(float)+4.03035218e-01, (float)+9.98598242e-01, (float)+6.48942364e-01, (float)+7.06741145e-01, (float)+2.74197278e-01, },
    {(float)+4.02340130e-01, (float)+9.98598242e-01, (float)+6.47972055e-01, (float)+7.06487241e-01, (float)+2.73338399e-01, },
    {(float)+4.01645042e-01, (float)+9.98598242e-01, (float)+6.47195808e-01, (float)+7.06233338e-01, (float)+2.72463002e-01, },
    {(float)+4.00949954e-01, (float)+9.98622000e-01, (float)+6.46419561e-01, (float)+7.05852482e-01, (float)+2.71604123e-01, },
    {(float)+4.00254866e-01, (float)+9.98622000e-01, (float)+6.45643315e-01, (float)+7.05598578e-01, (float)+2.70745243e-01, },
    {(float)+3.99559778e-01, (float)+9.98622000e-01, (float)+6.44867068e-01, (float)+7.05217722e-01, (float)+2.69886364e-01, },
    {(float)+3.98864690e-01, (float)+9.98645759e-01, (float)+6.43896759e-01, (float)+7.04963819e-01, (float)+2.69027484e-01, },
    {(float)+3.98169601e-01, (float)+9.98645759e-01, (float)+6.43120512e-01, (float)+7.04709915e-01, (float)+2.68168605e-01, },
    {(float)+3.97474513e-01, (float)+9.98645759e-01, (float)+6.42344265e-01, (float)+7.04329059e-01, (float)+2.67309725e-01, },
    {(float)+3.96779425e-01, (float)+9.98669518e-01, (float)+6.41568019e-01, (float)+7.04075156e-01, (float)+2.66434329e-01, },
    {(float)+3.96084337e-01, (float)+9.98669518e-01, (float)+6.40791772e-01, (float)+7.03694300e-01, (float)+2.65575449e-01, },
    {(float)+3.95389249e-01, (float)+9.98669518e-01, (float)+6.40015525e-01, (float)+7.03440396e-01, (float)+2.64716570e-01, },
    {(float)+3.94694161e-01, (float)+9.98693276e-01, (float)+6.39045216e-01, (float)+7.03186492e-01, (float)+2.63857690e-01, },
    {(float)+3.93999073e-01, (float)+9.98693276e-01, (float)+6.38268970e-01, (float)+7.02805637e-01, (float)+2.62998811e-01, },
    {(float)+3.93303985e-01, (float)+9.98693276e-01, (float)+6.37492723e-01, (float)+7.02551733e-01, (float)+2.62139931e-01, },
    {(float)+3.92608897e-01, (float)+9.98717035e-01, (float)+6.36716476e-01, (float)+7.02297829e-01, (float)+2.61281052e-01, },
    {(float)+3.91913809e-01, (float)+9.98717035e-01, (float)+6.35940229e-01, (float)+7.01916973e-01, (float)+2.60405655e-01, },
    {(float)+3.91218721e-01, (float)+9.98717035e-01, (float)+6.34969920e-01, (float)+7.01663070e-01, (float)+2.59546776e-01, },
    {(float)+3.90523633e-01, (float)+9.98740794e-01, (float)+6.34193674e-01, (float)+7.01282214e-01, (float)+2.58687896e-01, },
    {(float)+3.89828545e-01, (float)+9.98740794e-01, (float)+6.33417427e-01, (float)+7.01028310e-01, (float)+2.57829017e-01, },
    {(float)+3.89133457e-01, (float)+9.98740794e-01, (float)+6.32641180e-01, (float)+7.00774406e-01, (float)+2.56970137e-01, },
    {(float)+3.88438369e-01, (float)+9.98764552e-01, (float)+6.31864933e-01, (float)+7.00393551e-01, (float)+2.56111258e-01, },
    {(float)+3.87743281e-01, (float)+9.98764552e-01, (float)+6.30894624e-01, (float)+7.00139647e-01, (float)+2.55252378e-01, },
    {(float)+3.87048193e-01, (float)+9.98764552e-01, (float)+6.30118378e-01, (float)+6.99758791e-01, (float)+2.54376982e-01, },
    {(float)+3.86353105e-01, (float)+9.98788311e-01, (float)+6.29342131e-01, (float)+6.99504888e-01, (float)+2.53518103e-01, },
    {(float)+3.85658017e-01, (float)+9.98788311e-01, (float)+6.28565884e-01, (float)+6.99250984e-01, (float)+2.52659223e-01, },
    {(float)+3.84962929e-01, (float)+9.98788311e-01, (float)+6.27789637e-01, (float)+6.98870128e-01, (float)+2.51800344e-01, },
    {(float)+3.84267841e-01, (float)+9.98812069e-01, (float)+6.26819329e-01, (float)+6.98616224e-01, (float)+2.50941464e-01, },
    {(float)+3.83572753e-01, (float)+9.98812069e-01, (float)+6.26043082e-01, (float)+6.98362321e-01, (float)+2.50082585e-01, },
    {(float)+3.82877665e-01, (float)+9.98812069e-01, (float)+6.25266835e-01, (float)+6.97981465e-01, (float)+2.49223705e-01, },
    {(float)+3.82182576e-01, (float)+9.98835828e-01, (float)+6.24490588e-01, (float)+6.97727561e-01, (float)+2.48348309e-01, },
    {(float)+3.81487488e-01, (float)+9.98835828e-01, (float)+6.23714341e-01, (float)+6.97346706e-01, (float)+2.47489429e-01, },
    {(float)+3.80792400e-01, (float)+9.98835828e-01, (float)+6.22744033e-01, (float)+6.97092802e-01, (float)+2.46630550e-01, },
    {(float)+3.80097312e-01, (float)+9.98859587e-01, (float)+6.21967786e-01, (float)+6.96838898e-01, (float)+2.45771670e-01, },
    {(float)+3.79402224e-01, (float)+9.98859587e-01, (float)+6.21191539e-01, (float)+6.96458042e-01, (float)+2.44912791e-01, },
    {(float)+3.78707136e-01, (float)+9.98859587e-01, (float)+6.20415292e-01, (float)+6.96204139e-01, (float)+2.44053911e-01, },
    {(float)+3.78012048e-01, (float)+9.98883345e-01, (float)+6.19639045e-01, (float)+6.95950235e-01, (float)+2.43195032e-01, },
    {(float)+3.77316960e-01, (float)+9.98883345e-01, (float)+6.18668737e-01, (float)+6.95569379e-01, (float)+2.42319635e-01, },
    {(float)+3.76621872e-01, (float)+9.98883345e-01, (float)+6.17892490e-01, (float)+6.95315475e-01, (float)+2.41460756e-01, },
    {(float)+3.75926784e-01, (float)+9.98907104e-01, (float)+6.17116243e-01, (float)+6.94934620e-01, (float)+2.40601876e-01, },
    {(float)+3.75231696e-01, (float)+9.98907104e-01, (float)+6.16339996e-01, (float)+6.94680716e-01, (float)+2.39742997e-01, },
    {(float)+3.74536608e-01, (float)+9.98907104e-01, (float)+6.15563749e-01, (float)+6.94426812e-01, (float)+2.38884117e-01, },
    {(float)+3.73841520e-01, (float)+9.98930862e-01, (float)+6.14593441e-01, (float)+6.94045957e-01, (float)+2.38025238e-01, },
    {(float)+3.73146432e-01, (float)+9.98930862e-01, (float)+6.13817194e-01, (float)+6.93792053e-01, (float)+2.37166358e-01, },
    {(float)+3.72451344e-01, (float)+9.98930862e-01, (float)+6.13040947e-01, (float)+6.93411197e-01, (float)+2.36307479e-01, },
    {(float)+3.71756256e-01, (float)+9.98954621e-01, (float)+6.12264700e-01, (float)+6.93157293e-01, (float)+2.35432082e-01, },
    {(float)+3.71061168e-01, (float)+9.98954621e-01, (float)+6.11488453e-01, (float)+6.92903390e-01, (float)+2.34573203e-01, },
    {(float)+3.70366080e-01, (float)+9.98954621e-01, (float)+6.10518145e-01, (float)+6.92522534e-01, (float)+2.33714323e-01, },
    {(float)+3.69670992e-01, (float)+9.98978380e-01, (float)+6.09741898e-01, (float)+6.92268630e-01, (float)+2.32855444e-01, },
    {(float)+3.68975904e-01, (float)+9.98978380e-01, (float)+6.08965651e-01, (float)+6.92014726e-01, (float)+2.31996564e-01, },
    {(float)+3.68280816e-01, (float)+9.99002138e-01, (float)+6.08189404e-01, (float)+6.91633871e-01, (float)+2.31137685e-01, },
    {(float)+3.67585728e-01, (float)+9.99002138e-01, (float)+6.07413157e-01, (float)+6.91379967e-01, (float)+2.30278805e-01, },
    {(float)+3.66890639e-01, (float)+9.99002138e-01, (float)+6.06442849e-01, (float)+6.90999111e-01, (float)+2.29403409e-01, },
    {(float)+3.66195551e-01, (float)+9.99025897e-01, (float)+6.05666602e-01, (float)+6.90745208e-01, (float)+2.28544530e-01, },
    {(float)+3.65500463e-01, (float)+9.99025897e-01, (float)+6.04890355e-01, (float)+6.90491304e-01, (float)+2.27685650e-01, },
    {(float)+3.64805375e-01, (float)+9.99025897e-01, (float)+6.04114108e-01, (float)+6.90110448e-01, (float)+2.26826771e-01, },
    {(float)+3.64110287e-01, (float)+9.99049656e-01, (float)+6.03337861e-01, (float)+6.89856544e-01, (float)+2.25967891e-01, },
    {(float)+3.63415199e-01, (float)+9.99049656e-01, (float)+6.02367553e-01, (float)+6.89475689e-01, (float)+2.25109012e-01, },
    {(float)+3.62720111e-01, (float)+9.99049656e-01, (float)+6.01591306e-01, (float)+6.89221785e-01, (float)+2.24250132e-01, },
    {(float)+3.62025023e-01, (float)+9.99073414e-01, (float)+6.00815059e-01, (float)+6.88967881e-01, (float)+2.23374736e-01, },
    {(float)+3.61329935e-01, (float)+9.99073414e-01, (float)+6.00038812e-01, (float)+6.88587026e-01, (float)+2.22515856e-01, },
    {(float)+3.60634847e-01, (float)+9.99073414e-01, (float)+5.99262565e-01, (float)+6.88333122e-01, (float)+2.21656977e-01, },
    {(float)+3.59939759e-01, (float)+9.99097173e-01, (float)+5.98292257e-01, (float)+6.88079218e-01, (float)+2.20798097e-01, },
    {(float)+3.59244671e-01, (float)+9.99097173e-01, (float)+5.97516010e-01, (float)+6.87698362e-01, (float)+2.19939218e-01, },
    {(float)+3.58433735e-01, (float)+9.99097173e-01, (float)+5.96739763e-01, (float)+6.87444459e-01, (float)+2.19080338e-01, },
    {(float)+3.57738647e-01, (float)+9.99120931e-01, (float)+5.95963516e-01, (float)+6.87063603e-01, (float)+2.18221459e-01, },
    {(float)+3.57043559e-01, (float)+9.99120931e-01, (float)+5.95187270e-01, (float)+6.86809699e-01, (float)+2.17346062e-01, },
    {(float)+3.56348471e-01, (float)+9.99120931e-01, (float)+5.94216961e-01, (float)+6.86555795e-01, (float)+2.16487183e-01, },
    {(float)+3.55653383e-01, (float)+9.99144690e-01, (float)+5.93440714e-01, (float)+6.86174940e-01, (float)+2.15628303e-01, },
    {(float)+3.54958295e-01, (float)+9.99144690e-01, (float)+5.92664467e-01, (float)+6.85921036e-01, (float)+2.14769424e-01, },
    {(float)+3.54263207e-01, (float)+9.99144690e-01, (float)+5.91888220e-01, (float)+6.85667132e-01, (float)+2.13910544e-01, },
    {(float)+3.53568119e-01, (float)+9.99168449e-01, (float)+5.91111974e-01, (float)+6.85286277e-01, (float)+2.13051665e-01, },
    {(float)+3.52873031e-01, (float)+9.99168449e-01, (float)+5.90141665e-01, (float)+6.85032373e-01, (float)+2.12192785e-01, },
    {(float)+3.52177943e-01, (float)+9.99168449e-01, (float)+5.89365418e-01, (float)+6.84651517e-01, (float)+2.11317389e-01, },
    {(float)+3.51482854e-01, (float)+9.99192207e-01, (float)+5.88589171e-01, (float)+6.84397613e-01, (float)+2.10458510e-01, },
    {(float)+3.50787766e-01, (float)+9.99192207e-01, (float)+5.87812925e-01, (float)+6.84143710e-01, (float)+2.09599630e-01, },
    {(float)+3.50092678e-01, (float)+9.99192207e-01, (float)+5.87036678e-01, (float)+6.83762854e-01, (float)+2.08740751e-01, },
    {(float)+3.49397590e-01, (float)+9.99215966e-01, (float)+5.86260431e-01, (float)+6.83508950e-01, (float)+2.07881871e-01, },
    {(float)+3.48702502e-01, (float)+9.99215966e-01, (float)+5.85290122e-01, (float)+6.83128094e-01, (float)+2.07022992e-01, },
    {(float)+3.48007414e-01, (float)+9.99215966e-01, (float)+5.84513875e-01, (float)+6.82874191e-01, (float)+2.06164112e-01, },
    {(float)+3.47312326e-01, (float)+9.99239724e-01, (float)+5.83737629e-01, (float)+6.82620287e-01, (float)+2.05288716e-01, },
    {(float)+3.46617238e-01, (float)+9.99239724e-01, (float)+5.82961382e-01, (float)+6.82239431e-01, (float)+2.04429836e-01, },
    {(float)+3.45922150e-01, (float)+9.99239724e-01, (float)+5.82185135e-01, (float)+6.81985527e-01, (float)+2.03570957e-01, },
    {(float)+3.45227062e-01, (float)+9.99263483e-01, (float)+5.81214826e-01, (float)+6.81731624e-01, (float)+2.02712077e-01, },
    {(float)+3.44531974e-01, (float)+9.99263483e-01, (float)+5.80438579e-01, (float)+6.81350768e-01, (float)+2.01853198e-01, },
    };
    vector<vector<float>> transpose_result = transposeMatrix(initial_data);
    cout << "转置结果：" << transpose_result.size() << " " << transpose_result[0].size() << endl;
    vector<vector<float>> input_data= transpose_result;

    if (input_data.size() != transpose_result.size())return -1;

    //vector<vector<float>> target = transpose_result;

    int function_type = 2;
    int node = 50;//第一层节点
    int laye2_neural_nodeCount = 20;






    vector<float> layer1_biasmatrix = {
    {(float)+1.68421641e-01, (float)+2.44911388e-02, (float)-1.31312773e-01, (float)+3.79722454e-02, (float)+2.31272325e-01, (float)-1.73957005e-01, (float)-1.17132843e-01, (float)-7.27598667e-02, (float)-2.92566922e-02, (float)-3.56956691e-01, (float)-4.30121273e-01, (float)+3.87377590e-01, (float)-2.66141266e-01, (float)+1.56898603e-01, (float)-2.47453451e-01, (float)-1.71110600e-01, (float)+6.60459101e-02, (float)+2.48885006e-02, (float)+4.87879403e-02, (float)+1.99593678e-01, (float)+1.23972878e-01, (float)-1.16563328e-01, (float)-1.44894093e-01, (float)-2.40382969e-01, (float)-5.26422203e-01, (float)+8.46071690e-02, (float)+1.31910712e-01, (float)+2.40750715e-01, (float)-1.57345727e-01, (float)-1.04145467e-01, (float)-2.71522462e-01, (float)-3.85118484e-01, (float)-6.09938562e-01, (float)-2.38799259e-01, (float)-3.66119519e-02, (float)-3.47802460e-01, (float)-1.59180090e-02, (float)+1.50297647e-02, (float)-1.21335298e-01, (float)+1.13030575e-01, (float)-3.31825078e-01, (float)-2.36397132e-01, (float)-1.67683557e-01, (float)-1.37425214e-01, (float)+1.94076955e-01, (float)+2.52836645e-01, (float)+1.68677568e-01, (float)-8.53091896e-01, (float)+1.71622232e-01, (float)-1.99485317e-01, },
    };
    vector<vector<float>> layel1_weightmatrix = {
    {(float)+2.87886888e-01, (float)+3.90110165e-01, (float)+2.55814761e-01, (float)-7.70412758e-02, (float)-1.59845695e-01, (float)+1.00470521e-01, (float)-1.23583414e-01, (float)-2.17254013e-01, (float)-6.13036118e-02, (float)-1.54288068e-01, (float)-2.23228008e-01, (float)-5.21492124e-01, (float)+2.46548541e-02, (float)-1.24169007e-01, (float)+1.62203610e-01, (float)+1.07050776e-01, (float)+1.20243512e-03, (float)+1.32140338e-01, (float)-2.24970251e-01, (float)-5.83082080e-01, (float)-2.45522454e-01, (float)-3.69904816e-01, (float)-4.00691420e-01, (float)-5.92573062e-02, (float)-1.02371864e-01, (float)+2.21142471e-01, (float)-9.57876265e-01, (float)-1.28700182e-01, (float)+3.13176215e-01, (float)+1.90682821e-02, (float)-4.77258205e-01, (float)-1.64299205e-01, (float)-3.37699264e-01, (float)-5.68770915e-02, (float)-2.71200910e-02, (float)-1.64501324e-01, (float)-2.18355190e-02, (float)-1.97718702e-02, (float)-1.20463260e-01, (float)+1.21873338e-02, (float)+5.86798489e-01, (float)-1.84255302e-01, (float)-9.21448991e-02, (float)+4.75968756e-02, (float)-3.44024539e-01, (float)-1.99734315e-01, (float)-1.99607566e-01, (float)-8.64112914e-01, (float)+7.64790103e-02, (float)-2.01234564e-01, },
    {(float)-3.48216176e-01, (float)-5.44499513e-03, (float)+1.28143921e-01, (float)+3.21895182e-02, (float)+4.22730327e-01, (float)+5.00158012e-01, (float)-2.89477646e-01, (float)+1.91804156e-01, (float)+3.32881302e-01, (float)+8.14183280e-02, (float)+3.07153583e-01, (float)+5.14467582e-02, (float)+2.24824458e-01, (float)+9.99265388e-02, (float)+2.14411542e-01, (float)+2.25768939e-01, (float)-5.07355750e-01, (float)-5.18253922e-01, (float)-1.58192560e-01, (float)-8.52717385e-02, (float)-2.00079188e-01, (float)-3.19638073e-01, (float)-3.60052496e-01, (float)-8.59328434e-02, (float)-4.92156208e-01, (float)+1.58290520e-01, (float)+1.37937397e-01, (float)+6.23261323e-03, (float)-2.55114019e-01, (float)+2.81947643e-01, (float)-3.11405540e-01, (float)+1.34980127e-01, (float)+8.08790386e-01, (float)-1.41020820e-01, (float)-1.50104091e-01, (float)+2.88148850e-01, (float)-3.47633749e-01, (float)+2.88199842e-01, (float)+1.06837638e-01, (float)+9.31175202e-02, (float)-1.69705003e-01, (float)+1.03955619e-01, (float)+1.91933259e-01, (float)-5.55969588e-02, (float)-3.38564754e-01, (float)+4.37728643e-01, (float)-1.05479799e-01, (float)-4.81359780e-01, (float)+3.17069411e-01, (float)+1.49282396e-01, },
    {(float)-1.83873214e-02, (float)-2.80707717e-01, (float)+1.23863876e-01, (float)-3.78164977e-01, (float)-1.63928360e-01, (float)-8.20855126e-02, (float)+3.94625127e-01, (float)+7.35121220e-02, (float)+9.44923460e-02, (float)+4.15275067e-01, (float)+9.32782888e-02, (float)-2.13887572e-01, (float)+2.95613825e-01, (float)+1.59060121e-01, (float)+1.34712636e-01, (float)+1.43044978e-01, (float)+3.30870822e-02, (float)+1.60215229e-01, (float)+2.35751495e-02, (float)-4.85163331e-02, (float)-1.39316440e-01, (float)+1.20162912e-01, (float)+4.64229099e-02, (float)+2.99060136e-01, (float)-3.67133617e-02, (float)-1.75083637e-01, (float)-6.11048341e-01, (float)-1.60871819e-02, (float)-2.50858404e-02, (float)+2.61618108e-01, (float)+2.23415658e-01, (float)+1.46389753e-01, (float)+1.08877532e-01, (float)-2.27777466e-01, (float)+1.94866493e-01, (float)+3.33008260e-01, (float)+2.25285828e-01, (float)+1.06226772e-01, (float)+2.03216240e-01, (float)-3.43166322e-01, (float)+3.28830332e-01, (float)+2.54228920e-01, (float)+1.73955873e-01, (float)-6.59229979e-02, (float)+2.96273511e-02, (float)-3.83667529e-01, (float)-3.21965247e-01, (float)-1.37968886e+00, (float)-9.81510710e-03, (float)+1.56524867e-01, },
    {(float)+7.45380148e-02, (float)-5.03469944e-01, (float)-7.02661052e-02, (float)-1.23614110e-01, (float)-2.22021267e-01, (float)-4.72181320e-01, (float)+4.90653008e-01, (float)-2.27739945e-01, (float)+6.12321384e-02, (float)+2.47046590e-01, (float)+2.57824451e-01, (float)+1.33767545e-01, (float)+1.13432825e-01, (float)+2.99501582e-04, (float)+9.89942253e-02, (float)-9.13952850e-03, (float)+2.41527140e-01, (float)-2.08013073e-01, (float)-2.33590931e-01, (float)+1.20978475e-01, (float)-1.70018867e-01, (float)+3.39797169e-01, (float)+1.40787661e-01, (float)+1.82925642e-01, (float)-2.62178153e-01, (float)-2.44695783e-01, (float)-2.03184307e-01, (float)-3.51190493e-02, (float)+5.62051721e-02, (float)+7.61202425e-02, (float)+2.89936155e-01, (float)+3.47697027e-02, (float)-2.31631696e-01, (float)+1.04874916e-01, (float)+1.94808260e-01, (float)-2.78064698e-01, (float)-2.21501246e-01, (float)+1.37362732e-02, (float)-2.27829650e-01, (float)+2.32300684e-01, (float)-5.23337126e-01, (float)+3.06435764e-01, (float)+3.26516718e-01, (float)+2.94782430e-01, (float)+5.46990693e-01, (float)+9.08779502e-02, (float)+2.49821514e-01, (float)-1.26148665e+00, (float)+2.95306128e-02, (float)-2.96160191e-01, },
    {(float)-2.15035692e-01, (float)+4.82458323e-01, (float)+4.31452086e-03, (float)+2.78559715e-01, (float)+2.27041751e-01, (float)-1.67791005e-02, (float)-3.17592651e-01, (float)-1.29125565e-01, (float)+1.48101915e-02, (float)-1.38790614e-03, (float)+2.80793279e-01, (float)+1.28899738e-01, (float)-5.94157577e-02, (float)-2.59969562e-01, (float)+7.78215453e-02, (float)+1.04676448e-01, (float)+1.84090972e-01, (float)+3.92030388e-01, (float)-7.99688473e-02, (float)+1.17084228e-01, (float)-5.06122634e-02, (float)+3.31082255e-01, (float)+5.58553278e-01, (float)+1.96201056e-01, (float)+1.37193546e-01, (float)-2.43973449e-01, (float)-9.65223253e-01, (float)-2.91055530e-01, (float)+1.43977866e-01, (float)-2.68448412e-01, (float)+4.75971848e-01, (float)+4.42193598e-01, (float)-3.37428242e-01, (float)+4.78451520e-01, (float)-3.03267002e-01, (float)+3.45874101e-01, (float)+4.41064656e-01, (float)+2.75925938e-02, (float)+2.01330334e-01, (float)-3.87982100e-01, (float)-1.80643812e-01, (float)-1.94675885e-02, (float)-2.39030838e-01, (float)+1.42028686e-02, (float)+4.63614240e-02, (float)-3.59107889e-02, (float)-1.07824199e-01, (float)-6.71592891e-01, (float)+3.88988286e-01, (float)+4.89139944e-01, },
    };








    vector<float> layer2_biasmatrix = {
    {(float)+1.73322171e-01, (float)-1.28905505e-01, (float)+7.65350163e-02, (float)-2.19228178e-01, (float)-2.35561896e-02, (float)-5.38888536e-02, (float)-3.97061527e-01, (float)+5.07393658e-01, (float)-9.09874588e-02, (float)-6.92246333e-02, (float)+3.97600904e-02, (float)+8.89922380e-02, (float)+3.16074491e-02, (float)+2.26830408e-01, (float)+1.81704998e-01, (float)+1.83452487e-01, (float)-8.76754671e-02, (float)-1.67065933e-02, (float)-1.05762273e-01, (float)-1.65680528e-01, },
    };
    vector<vector<float>> layel2_weightmatrix = {
    {(float)+4.73856807e-01, (float)-1.03001229e-01, (float)+1.18264258e-01, (float)+3.49980086e-01, (float)-3.81941795e-02, (float)-1.79504871e-01, (float)+7.53153861e-02, (float)+7.97541320e-01, (float)-1.49157755e-02, (float)-4.35582697e-02, (float)+3.68395865e-01, (float)-1.90903261e-01, (float)+4.53553975e-01, (float)+1.15082279e-01, (float)+6.98810443e-02, (float)-1.53582588e-01, (float)-1.53938040e-01, (float)-5.54033443e-02, (float)-2.31063262e-01, (float)+1.98608160e-01, },
    {(float)+2.69983470e-01, (float)+3.54573846e-01, (float)+6.73857749e-01, (float)-1.37243345e-01, (float)+5.44838548e-01, (float)-6.32181823e-01, (float)+5.76390564e-01, (float)-1.33221936e+00, (float)-1.36985213e-01, (float)+4.30339545e-01, (float)-2.03202233e-01, (float)+2.20386595e-01, (float)-1.31530270e-01, (float)+5.79647839e-01, (float)+4.52516317e-01, (float)+5.81084251e-01, (float)+4.64046061e-01, (float)-3.91483121e-02, (float)-5.79720616e-01, (float)+2.41778433e-01, },
    {(float)-4.51453738e-02, (float)-3.05666655e-01, (float)+3.76019508e-01, (float)+1.81281775e-01, (float)+2.52025247e-01, (float)+2.66813040e-01, (float)+3.23199600e-01, (float)-2.74656773e-01, (float)+2.68599130e-02, (float)+8.11671242e-02, (float)+1.39004318e-02, (float)+2.86192864e-01, (float)+3.30972344e-01, (float)-1.72301382e-01, (float)+6.23200759e-02, (float)+1.93020552e-01, (float)+2.52981454e-01, (float)+1.65153503e-01, (float)-1.02381170e-01, (float)-5.10948002e-02, },
    {(float)+2.42475912e-01, (float)+3.88765514e-01, (float)+2.14686319e-01, (float)-5.35177998e-02, (float)+1.08811043e-01, (float)-5.49108684e-02, (float)-4.77893054e-01, (float)-6.16557419e-01, (float)-4.15936887e-01, (float)+3.69141489e-01, (float)-2.55347580e-01, (float)-2.74839461e-01, (float)-5.32790244e-01, (float)+1.08058415e-01, (float)-3.00618351e-01, (float)+6.92506656e-02, (float)+3.06567140e-02, (float)-5.96883520e-02, (float)-6.54279590e-01, (float)+5.44447452e-02, },
    {(float)-2.81553090e-01, (float)-1.51747212e-01, (float)+2.91170359e-01, (float)-3.02819341e-01, (float)-2.46430144e-01, (float)+1.51950717e-01, (float)-1.71429956e+00, (float)+1.42779604e-01, (float)-5.17898560e-01, (float)+6.74099401e-02, (float)+3.34388837e-02, (float)-4.48661111e-02, (float)-1.93880960e-01, (float)+3.74339581e-01, (float)-9.99987721e-02, (float)+4.56120849e-01, (float)+3.29039484e-01, (float)-9.93021205e-02, (float)-1.34005740e-01, (float)-1.73560567e-02, },
    {(float)-6.17351830e-01, (float)+3.17955941e-01, (float)+2.17330724e-01, (float)+5.83876576e-03, (float)+1.06551483e-01, (float)+4.36838754e-02, (float)+1.42597646e-01, (float)-3.17116529e-01, (float)+9.80939567e-02, (float)+2.85604447e-01, (float)-2.11191662e-02, (float)+3.57531309e-02, (float)+1.22805998e-01, (float)-3.94655406e-01, (float)-1.76669937e-02, (float)+1.71028271e-01, (float)+3.34799469e-01, (float)-3.04922462e-01, (float)-3.33829492e-01, (float)+3.65666568e-01, },
    {(float)+5.53773940e-01, (float)-1.16586715e-01, (float)-2.32059911e-01, (float)+2.71077573e-01, (float)-2.79516280e-01, (float)+3.30918252e-01, (float)-2.56906629e-01, (float)+8.13155651e-01, (float)+3.66437733e-01, (float)-3.28856885e-01, (float)+4.39302832e-01, (float)-1.01277493e-01, (float)+3.57942253e-01, (float)-3.11986089e-01, (float)-4.37251151e-01, (float)-4.34311479e-01, (float)-1.23395838e-01, (float)+2.38434985e-01, (float)+4.31929320e-01, (float)-2.04175726e-01, },
    {(float)-5.04355133e-01, (float)+3.61227453e-01, (float)-2.25167796e-01, (float)-1.84805050e-01, (float)+1.93542875e-02, (float)+1.21388063e-01, (float)-3.99042591e-02, (float)-3.71643156e-01, (float)-1.94673955e-01, (float)+3.79752070e-01, (float)+2.85868887e-02, (float)+1.02294698e-01, (float)+2.20597424e-02, (float)+4.85616960e-02, (float)-1.19111329e-01, (float)-1.47079185e-01, (float)+4.39861491e-02, (float)-3.71347554e-02, (float)+1.72524676e-01, (float)+2.55917698e-01, },
    {(float)-3.50569449e-02, (float)+9.87679437e-02, (float)-1.79669615e-02, (float)+4.55662161e-02, (float)-1.02287449e-01, (float)+2.57044196e-01, (float)-5.94777390e-02, (float)+9.42976922e-02, (float)+1.66029371e-02, (float)-1.78215265e-01, (float)+1.12245359e-01, (float)+2.87656546e-01, (float)-2.47168899e-01, (float)+1.32713974e-01, (float)+1.86384186e-01, (float)-8.59196186e-02, (float)-2.31403008e-01, (float)+2.54439890e-01, (float)-1.71931796e-02, (float)-2.69823194e-01, },
    {(float)-1.36917129e-01, (float)-2.04738583e-02, (float)-3.59264433e-01, (float)+4.46125329e-01, (float)-3.68903786e-01, (float)+5.53502500e-01, (float)+5.88977098e-01, (float)-1.06930399e+00, (float)+2.19181292e-02, (float)-4.33482409e-01, (float)+3.17544162e-01, (float)-5.37898438e-03, (float)+4.15654667e-02, (float)-6.65422142e-01, (float)+1.97253749e-01, (float)-6.52943552e-01, (float)+2.10474148e-01, (float)-1.33175347e-02, (float)+9.57638681e-01, (float)+4.96341325e-02, },
    {(float)-2.95737743e-01, (float)-3.73072386e-01, (float)-1.04605548e-01, (float)+3.21072459e-01, (float)-5.72764277e-01, (float)+7.31196642e-01, (float)-2.44487926e-01, (float)-1.84389389e+00, (float)-4.80257839e-01, (float)-2.82314092e-01, (float)+8.06103200e-02, (float)+3.33221443e-02, (float)-6.00595534e-01, (float)-7.40292728e-01, (float)-1.50614709e-01, (float)-6.52074754e-01, (float)+5.21318734e-01, (float)+3.42487484e-01, (float)+8.08682740e-01, (float)-2.50500649e-01, },
    {(float)+1.91543654e-01, (float)-3.28906775e-01, (float)-1.78471599e-02, (float)-5.80671489e-01, (float)-3.00412416e-01, (float)-1.98560357e-01, (float)-3.13617229e+00, (float)+9.29527640e-01, (float)-6.06240630e-01, (float)+1.91138387e-02, (float)-1.89334869e-01, (float)-6.43222034e-02, (float)-2.04807565e-01, (float)+2.83787042e-01, (float)+3.30390990e-01, (float)-9.27706715e-03, (float)-3.85386348e-01, (float)+2.26789221e-01, (float)+2.08888669e-02, (float)+1.62060231e-01, },
    {(float)-2.84660667e-01, (float)-3.68025690e-01, (float)-1.51040956e-01, (float)+2.53473580e-01, (float)-8.70976523e-02, (float)+2.16960564e-01, (float)+6.65274620e-01, (float)-4.23974484e-01, (float)-1.57950267e-01, (float)-3.73995900e-01, (float)-5.79479057e-03, (float)+2.04320133e-01, (float)+3.20736498e-01, (float)-3.59830678e-01, (float)+2.36976668e-01, (float)-2.23865122e-01, (float)+3.28201443e-01, (float)-6.38026595e-02, (float)+3.23829949e-01, (float)-1.57190580e-02, },
    {(float)-1.46228626e-01, (float)-4.18874025e-01, (float)+7.64681846e-02, (float)-1.13904424e-01, (float)-2.45818406e-01, (float)-9.27553549e-02, (float)-5.23785830e-01, (float)+8.09709907e-01, (float)+3.68373888e-03, (float)-1.56423450e-01, (float)+1.07817389e-01, (float)-7.89186284e-02, (float)+1.94292501e-01, (float)-2.64711052e-01, (float)+2.41959512e-01, (float)-2.46326085e-02, (float)-4.40436117e-02, (float)-2.04932421e-01, (float)-6.74491236e-03, (float)-1.07277274e-01, },
    {(float)-6.68839067e-02, (float)+6.43469542e-02, (float)+7.38839805e-02, (float)+3.14904511e-01, (float)+2.43181805e-03, (float)+2.49936566e-01, (float)+5.06574273e-01, (float)-4.44733292e-01, (float)+1.65750610e-03, (float)-2.03499064e-01, (float)+9.12165269e-02, (float)+2.93568671e-01, (float)+3.08469143e-02, (float)-4.50966954e-01, (float)+2.93491870e-01, (float)+2.01496221e-02, (float)+1.55450463e-01, (float)+1.01716243e-01, (float)+4.82681662e-01, (float)-1.38745889e-01, },
    {(float)-2.03686923e-01, (float)+8.57092515e-02, (float)+3.93282287e-02, (float)+2.24058945e-02, (float)+1.99496523e-01, (float)-1.41512752e-01, (float)+1.95769612e-02, (float)-8.92945230e-02, (float)-2.10176185e-01, (float)-8.82888585e-02, (float)+2.47508496e-01, (float)-1.03623100e-01, (float)+9.82658267e-02, (float)-3.14887851e-01, (float)-4.97110635e-02, (float)+5.41659482e-02, (float)+3.77572119e-01, (float)-2.10466668e-01, (float)+3.72918516e-01, (float)-1.67834178e-01, },
    {(float)+9.49905694e-01, (float)+1.71949729e-01, (float)-2.28336096e-01, (float)-9.02040899e-02, (float)-2.78603405e-01, (float)-2.91660488e-01, (float)+7.82121271e-02, (float)+1.13746867e-01, (float)+3.72758359e-01, (float)-9.10637341e-03, (float)+6.45654202e-01, (float)+3.30893844e-02, (float)+3.10358495e-01, (float)+4.65274990e-01, (float)-5.83722442e-02, (float)-3.81216228e-01, (float)-1.20916940e-01, (float)-8.20587426e-02, (float)+3.06891967e-02, (float)+4.44262708e-03, },
    {(float)+6.06410265e-01, (float)-1.41090244e-01, (float)+2.32571244e-01, (float)+1.20270602e-01, (float)-6.38714712e-03, (float)-2.99463600e-01, (float)+8.97098929e-02, (float)-2.80912727e-01, (float)+2.80637980e-01, (float)+1.45673215e-01, (float)+6.20964989e-02, (float)-1.79187451e-02, (float)-1.07462600e-01, (float)+3.25038552e-01, (float)-2.56615430e-01, (float)-6.36441782e-02, (float)+1.52273029e-01, (float)-8.01354721e-02, (float)+8.65308270e-02, (float)+1.06872924e-01, },
    {(float)+1.92916822e-02, (float)+2.16112226e-01, (float)-2.06868067e-01, (float)+2.53351638e-03, (float)+2.57761925e-01, (float)-1.67662218e-01, (float)+2.18870282e-01, (float)-1.90250531e-01, (float)+1.34640962e-01, (float)+2.43745670e-01, (float)+7.26924539e-02, (float)-3.14644910e-02, (float)-2.00519189e-01, (float)+1.89328238e-01, (float)-3.71168070e-02, (float)+2.57467151e-01, (float)-2.86023855e-01, (float)-1.53711185e-01, (float)-2.69727677e-01, (float)+3.15336645e-01, },
    {(float)+5.56478739e-01, (float)-5.95872141e-02, (float)-7.26500928e-01, (float)-9.16945517e-01, (float)-9.65991735e-01, (float)-1.05664968e-01, (float)-8.62944007e-01, (float)+6.11543655e-01, (float)-3.98626685e-01, (float)-6.54390007e-02, (float)-4.62344550e-02, (float)+1.12000480e-01, (float)-8.48714292e-01, (float)+1.78866908e-01, (float)-5.29064685e-02, (float)-5.40744066e-02, (float)-3.81010622e-01, (float)+2.91137725e-01, (float)-4.71473634e-01, (float)+3.48429382e-01, },
    {(float)+3.56267728e-02, (float)+1.64356083e-01, (float)-4.64429632e-02, (float)-2.79868424e-01, (float)-1.97604865e-01, (float)-1.02462836e-01, (float)+2.40246952e-01, (float)-9.95911807e-02, (float)-3.79435942e-02, (float)+2.44918332e-01, (float)-1.39120877e-01, (float)-2.03026071e-01, (float)-7.46201500e-02, (float)-3.60869355e-02, (float)-1.63424939e-01, (float)+3.01335722e-01, (float)-2.99533188e-01, (float)+1.76605582e-01, (float)-1.17759109e-01, (float)+2.23302200e-01, },
    {(float)+3.84299815e-01, (float)-3.11305046e-01, (float)-2.53311276e-01, (float)+4.83407266e-02, (float)-2.20567048e-01, (float)-2.43343562e-02, (float)-2.31644988e-01, (float)-4.42105532e-01, (float)+1.28773898e-01, (float)-3.29530835e-02, (float)+6.66011155e-01, (float)+4.48566787e-02, (float)-5.05460620e-01, (float)-5.40303625e-02, (float)+8.65355656e-02, (float)-3.77778977e-01, (float)-1.60021573e-01, (float)+3.14372808e-01, (float)+3.51820104e-02, (float)+2.77384575e-02, },
    {(float)+7.59952068e-01, (float)-3.13244052e-02, (float)-3.47193897e-01, (float)-1.45078406e-01, (float)-4.67173696e-01, (float)-3.76375049e-01, (float)-2.71952599e-01, (float)-1.57639575e+00, (float)-1.22666441e-01, (float)-1.61458313e-01, (float)+3.99836898e-01, (float)-1.17272720e-01, (float)-6.16070330e-01, (float)+3.34288836e-01, (float)-1.57162249e-01, (float)-7.04805672e-01, (float)+2.18364313e-01, (float)+1.09411784e-01, (float)+2.31009781e-01, (float)+3.08173716e-01, },
    {(float)+3.02364439e-01, (float)-1.41324461e-01, (float)-1.44702783e-02, (float)+2.75620073e-01, (float)+1.45628005e-01, (float)+2.63231128e-01, (float)-8.56043547e-02, (float)-1.00597687e-01, (float)+1.16344541e-01, (float)-6.70530228e-03, (float)+2.38479972e-01, (float)-8.99940059e-02, (float)+1.02300271e-01, (float)-4.57402915e-01, (float)+1.83004722e-01, (float)-4.21214789e-01, (float)+3.94072831e-01, (float)+2.84489334e-01, (float)+5.24820864e-01, (float)+1.56035751e-01, },
    {(float)-1.33156687e-01, (float)+1.31750613e-01, (float)+2.69194953e-02, (float)-2.12247908e-01, (float)-1.63947746e-01, (float)+5.54300584e-02, (float)+8.58506858e-01, (float)-7.42078483e-01, (float)+5.75233735e-02, (float)+1.73541665e-01, (float)-1.26281187e-01, (float)-5.19875959e-02, (float)-7.51430765e-02, (float)-3.09845090e-01, (float)-1.73910469e-01, (float)-3.99813116e-01, (float)-1.20099887e-01, (float)-1.12285800e-01, (float)+1.16020210e-01, (float)+2.95548946e-01, },
    {(float)-2.78123319e-01, (float)+1.93885952e-01, (float)+1.95507139e-01, (float)-3.04409832e-01, (float)+4.02105659e-01, (float)-2.14455187e-01, (float)+4.54742402e-01, (float)+1.61676586e-01, (float)-2.00021029e-01, (float)+3.91521245e-01, (float)-3.92875850e-01, (float)-9.96950939e-02, (float)+4.11245167e-01, (float)-9.67818871e-02, (float)-2.74415702e-01, (float)+3.91646385e-01, (float)+1.40247911e-01, (float)+9.79138631e-03, (float)-1.57886207e-01, (float)-1.52840033e-01, },
    {(float)-2.04527333e-01, (float)+3.93421680e-01, (float)+2.20101997e-01, (float)+3.54944348e-01, (float)-4.54277545e-02, (float)+1.72806516e-01, (float)+3.44325066e-01, (float)-3.26797932e-01, (float)+1.72735184e-01, (float)-2.57997680e-02, (float)+4.11332659e-02, (float)-9.75884497e-02, (float)-1.49120241e-01, (float)-4.52529371e-01, (float)+3.04518133e-01, (float)-1.69541478e-01, (float)-1.92398742e-01, (float)+1.81110412e-01, (float)+3.05000357e-02, (float)-7.32688233e-02, },
    {(float)+2.95005068e-02, (float)-3.62312533e-02, (float)-7.82620013e-02, (float)+1.33536547e-01, (float)+7.18655586e-02, (float)-1.46350250e-01, (float)-1.40519008e-01, (float)+5.81943095e-01, (float)-1.07856691e-01, (float)-3.69208865e-02, (float)-5.23943715e-02, (float)-1.43098220e-01, (float)-1.25525057e-01, (float)+3.14513236e-01, (float)-2.43380159e-01, (float)+1.56651437e-01, (float)-3.03654015e-01, (float)+3.01543295e-01, (float)+3.05150282e-02, (float)-2.47923076e-01, },
    {(float)+2.39072889e-01, (float)+1.20839700e-01, (float)+4.50878218e-02, (float)+2.28929833e-01, (float)+2.41853178e-01, (float)-9.29381251e-02, (float)+3.82238626e-01, (float)-7.35009789e-01, (float)+4.24828976e-01, (float)-2.42048398e-01, (float)-6.92678094e-02, (float)+2.95589298e-01, (float)+4.03158486e-01, (float)-8.91505368e-03, (float)+2.50100363e-02, (float)-2.37961218e-01, (float)-1.14069484e-01, (float)+2.16984883e-01, (float)+4.42574004e-04, (float)-3.69153172e-02, },
    {(float)+9.83625427e-02, (float)-1.06701069e-01, (float)+9.59209725e-02, (float)+3.27678531e-01, (float)-8.65366161e-02, (float)+4.34031695e-01, (float)+1.00446224e+00, (float)+4.14590478e-01, (float)-6.50332049e-02, (float)+3.40882763e-02, (float)+6.01175055e-02, (float)+1.47400215e-01, (float)-1.54331326e-02, (float)-5.48733413e-01, (float)+2.84378350e-01, (float)+7.99115151e-02, (float)+6.04964187e-03, (float)-1.81736827e-01, (float)+2.79375553e-01, (float)+1.84085930e-03, },
    {(float)+1.01092827e+00, (float)+2.49057040e-01, (float)-1.67608842e-01, (float)+7.45267794e-02, (float)-9.75686431e-01, (float)+1.19563274e-01, (float)-5.69944143e-01, (float)-1.30429924e+00, (float)+8.30752254e-02, (float)-2.31499106e-01, (float)+5.39671183e-01, (float)-9.50129777e-02, (float)-2.13227183e-01, (float)-1.96348473e-01, (float)-3.94184381e-01, (float)-7.89835751e-01, (float)+2.61283606e-01, (float)+3.86539191e-01, (float)+2.19297364e-01, (float)+2.39409968e-01, },
    {(float)-4.63788778e-01, (float)-4.72269068e-03, (float)+1.20393813e-01, (float)-1.09281987e-02, (float)-4.83293563e-01, (float)+3.24798375e-02, (float)+7.01906860e-01, (float)-2.26703382e+00, (float)+3.10523003e-01, (float)-1.81207433e-02, (float)+4.32826459e-01, (float)+1.76645115e-01, (float)-6.40796959e-01, (float)-5.12784302e-01, (float)-2.02958304e-02, (float)-6.93210602e-01, (float)+6.87700450e-01, (float)-5.26606776e-02, (float)+8.07354987e-01, (float)+2.77725756e-01, },
    {(float)-8.27127397e-01, (float)+2.05712706e-01, (float)-1.45934075e-01, (float)+3.78252983e-01, (float)+8.21772143e-02, (float)-5.65002119e-06, (float)+2.11365491e-01, (float)-1.98005483e-01, (float)-1.86982200e-01, (float)+3.33867490e-01, (float)-2.39875406e-01, (float)+6.15306124e-02, (float)-2.48389825e-01, (float)-4.87385929e-01, (float)-2.00077653e-01, (float)-4.05092001e-01, (float)+1.94434628e-01, (float)+3.77704799e-01, (float)-3.21991779e-02, (float)+3.48044306e-01, },
    {(float)-2.06138473e-02, (float)+3.00257236e-01, (float)-4.27270643e-02, (float)-2.15584829e-01, (float)-2.64927924e-01, (float)-2.70633101e-01, (float)+1.09264247e-01, (float)-1.11451626e+00, (float)+2.81667650e-01, (float)+2.64471192e-02, (float)+6.70246175e-03, (float)-7.86806643e-02, (float)-3.78837794e-01, (float)-9.16051641e-02, (float)-2.17605278e-01, (float)-1.99248102e-02, (float)+2.04349503e-01, (float)+1.15134746e-01, (float)+1.99460592e-02, (float)+1.62425265e-01, },
    {(float)+7.28615597e-02, (float)+6.70136139e-02, (float)-3.09315383e-01, (float)+2.24984273e-01, (float)+1.00675255e-01, (float)+2.60187383e-03, (float)+2.87340909e-01, (float)+2.11119831e-01, (float)+1.45500928e-01, (float)-1.92200467e-01, (float)+5.16275316e-02, (float)-2.46927544e-01, (float)+2.37333134e-01, (float)+1.47449356e-02, (float)-2.47074351e-01, (float)-3.41108710e-01, (float)-7.88824782e-02, (float)-2.09792890e-02, (float)+8.33001584e-02, (float)-1.63459420e-01, },
    {(float)-3.02889705e-01, (float)-8.42571259e-02, (float)+4.88151848e-01, (float)+2.19562352e-01, (float)+1.98666304e-01, (float)+2.09364370e-01, (float)+2.39082909e+00, (float)-2.37535501e+00, (float)+1.59202158e-01, (float)-1.25104785e-01, (float)+1.08064413e-01, (float)-1.44867569e-01, (float)-4.52051908e-01, (float)-5.25168717e-01, (float)+2.30626807e-01, (float)-3.03123951e-01, (float)+1.92538410e-01, (float)-1.95980191e-01, (float)+6.83349788e-01, (float)+3.05526674e-01, },
    {(float)+6.84074879e-01, (float)+3.39121759e-01, (float)+1.77426487e-01, (float)+2.24953786e-01, (float)-2.28763282e-01, (float)-2.46492863e-01, (float)+3.45913559e-01, (float)-8.41705382e-01, (float)-1.86479725e-02, (float)-1.68041691e-01, (float)+3.90442163e-01, (float)-4.31089289e-03, (float)-1.47950530e-01, (float)+4.04490054e-01, (float)-8.25328752e-02, (float)+9.87806320e-02, (float)-4.93352376e-02, (float)+5.12091480e-02, (float)-8.73779058e-02, (float)-5.13212308e-02, },
    {(float)-2.40258932e-01, (float)-2.64168888e-01, (float)-1.34229183e-01, (float)-1.50127366e-01, (float)+1.17585801e-01, (float)+4.01567928e-02, (float)-2.12854683e-01, (float)+3.18748653e-01, (float)-1.39218137e-01, (float)-1.81036785e-01, (float)-1.68088108e-01, (float)+3.42119098e-01, (float)-2.13434160e-01, (float)-3.52497458e-01, (float)+3.60688537e-01, (float)+1.87813371e-01, (float)+1.62317693e-01, (float)+1.95157945e-01, (float)+2.40616240e-02, (float)-2.47618511e-01, },
    {(float)+9.86938365e-03, (float)+3.35829854e-01, (float)+1.40721098e-01, (float)-1.67492837e-01, (float)-1.94742996e-02, (float)+1.39488250e-01, (float)+7.38655210e-01, (float)-1.08048344e+00, (float)+2.90591508e-01, (float)-1.63935885e-01, (float)-2.81995721e-02, (float)-1.43875629e-01, (float)+8.13623518e-02, (float)-1.16707072e-01, (float)-1.43026754e-01, (float)+5.40372878e-02, (float)+6.94525242e-02, (float)-6.79438561e-02, (float)+2.62111068e-01, (float)+1.56062424e-01, },
    {(float)-1.15314260e-01, (float)+1.68765977e-01, (float)-3.45439255e-01, (float)-1.56683072e-01, (float)-9.36535820e-02, (float)+1.22718073e-01, (float)-2.66158551e-01, (float)+1.02164984e+00, (float)-1.28726676e-01, (float)+2.05204174e-01, (float)-1.31604925e-01, (float)-9.18603688e-02, (float)-1.51825368e-01, (float)-2.64303565e-01, (float)-2.65177637e-01, (float)+8.02232102e-02, (float)-5.84374189e-01, (float)-6.37075603e-02, (float)-3.65068853e-01, (float)+2.41411656e-01, },
    {(float)-1.74350336e-01, (float)+1.32879645e-01, (float)+2.15736806e-01, (float)+1.55437306e-01, (float)+7.73060083e-01, (float)-1.76752731e-01, (float)+2.76515532e+00, (float)-5.90484381e-01, (float)+5.99845886e-01, (float)+8.14563259e-02, (float)+2.12459147e-01, (float)+2.94250071e-01, (float)+1.76324964e-01, (float)-4.75801915e-01, (float)-1.61537856e-01, (float)+1.98822543e-01, (float)+3.79377544e-01, (float)+2.34634499e-03, (float)-4.30913456e-02, (float)+4.33942527e-02, },
    {(float)-6.11464307e-02, (float)-5.80781877e-01, (float)-7.80367106e-02, (float)+1.66825593e-01, (float)-1.85398862e-01, (float)+4.70928878e-01, (float)-3.24117035e-01, (float)-1.52652845e-01, (float)+5.18227890e-02, (float)-2.21964031e-01, (float)+4.08912003e-01, (float)-1.88145101e-01, (float)-5.41767441e-02, (float)-4.64127928e-01, (float)+7.03215674e-02, (float)-5.15334427e-01, (float)-3.23285125e-02, (float)-9.11264643e-02, (float)+4.41561610e-01, (float)-2.83401698e-01, },
    {(float)-1.31048217e-01, (float)-2.74964690e-01, (float)-1.23332329e-01, (float)+1.22075014e-01, (float)-1.78797305e-01, (float)+2.46183097e-01, (float)-5.37162423e-01, (float)+2.19819173e-01, (float)-4.86575365e-02, (float)+1.25376761e-01, (float)+5.76175749e-02, (float)+6.49361759e-02, (float)+1.58170566e-01, (float)-2.65598714e-01, (float)-1.96848467e-01, (float)-3.70155424e-01, (float)+2.31924102e-01, (float)-7.08969906e-02, (float)+4.58353341e-01, (float)-3.22711617e-01, },
    {(float)+1.18617319e-01, (float)+6.45723864e-02, (float)+1.41985759e-01, (float)+2.46368279e-03, (float)-5.35511784e-03, (float)+1.71189114e-01, (float)-1.87283665e-01, (float)+1.69767335e-01, (float)-1.52389873e-02, (float)+1.04495905e-01, (float)+3.19407731e-01, (float)-1.68264046e-01, (float)+1.35668233e-01, (float)-2.10245371e-01, (float)+1.36555079e-02, (float)-2.31030539e-01, (float)+7.28457198e-02, (float)+2.76296698e-02, (float)-8.98002982e-02, (float)-5.19037694e-02, },
    {(float)+2.42593616e-01, (float)-3.25373620e-01, (float)-1.75000113e-02, (float)-2.40288168e-01, (float)-5.82701027e-01, (float)-1.35152519e-01, (float)-9.43432271e-01, (float)+7.32420862e-01, (float)-2.35001847e-01, (float)-4.86309975e-01, (float)+4.56376076e-01, (float)+3.23449939e-01, (float)+3.75867961e-03, (float)-2.12248459e-01, (float)+3.81341666e-01, (float)-3.67431223e-01, (float)-3.74204695e-01, (float)-1.27315968e-02, (float)+2.36479659e-02, (float)+1.04468532e-01, },
    {(float)-3.03562015e-01, (float)-2.47496754e-01, (float)+1.59538940e-01, (float)-2.48575002e-01, (float)-4.00320590e-01, (float)-2.24631280e-03, (float)-1.24422145e+00, (float)+7.13605046e-01, (float)-6.30391896e-01, (float)-6.28697872e-02, (float)-3.72815520e-01, (float)-1.67302608e-01, (float)-2.36112654e-01, (float)+1.67714849e-01, (float)-1.08767487e-01, (float)+4.47448879e-01, (float)-3.70775461e-01, (float)+1.17135108e-01, (float)-1.43293917e-01, (float)-1.84637278e-01, },
    {(float)-3.87854800e-02, (float)-9.47453901e-02, (float)+1.25517771e-02, (float)-3.07605356e-01, (float)+1.76802456e-01, (float)-2.54368097e-01, (float)-4.34930593e-01, (float)+1.11409016e-01, (float)-3.33357692e-01, (float)+1.83857054e-01, (float)-4.09473330e-02, (float)-7.68284798e-02, (float)-9.08874646e-02, (float)+1.36975031e-02, (float)-3.28192621e-01, (float)+2.09378675e-01, (float)-1.12935111e-01, (float)-1.53550953e-01, (float)-2.72104263e-01, (float)+1.66463330e-01, },
    {(float)-3.98238510e-01, (float)+3.92864585e-01, (float)-3.21742356e-01, (float)+1.17261320e-01, (float)+1.10504575e-01, (float)+1.95862427e-01, (float)+5.08987248e-01, (float)-4.70603585e-01, (float)+2.82384634e-01, (float)+8.07489902e-02, (float)+8.73940811e-02, (float)+1.35639682e-01, (float)-1.61580279e-01, (float)-1.66833669e-01, (float)-1.17111832e-01, (float)-1.13999248e-01, (float)+3.11272025e-01, (float)-2.36636829e-02, (float)-2.11394075e-02, (float)+7.49119818e-02, },
    {(float)+3.26750278e-01, (float)-4.04228359e-01, (float)+3.31953466e-01, (float)-1.48772486e-02, (float)-2.48020530e-01, (float)-2.18478531e-01, (float)-3.73292357e-01, (float)-7.71178752e-02, (float)+5.59197627e-02, (float)-2.61889488e-01, (float)-1.15832083e-01, (float)+8.03924352e-02, (float)-1.64010689e-01, (float)+1.26837596e-01, (float)+2.71974802e-01, (float)+3.00273299e-01, (float)-8.75754189e-03, (float)+1.34604663e-01, (float)+5.86064681e-02, (float)+5.40536419e-02, },
    {(float)-1.80200279e-01, (float)-1.05129164e-02, (float)+4.86949056e-01, (float)-1.52058408e-01, (float)-2.13145584e-01, (float)-5.46483211e-02, (float)+1.69983447e-01, (float)-1.90136278e+00, (float)-2.92850044e-02, (float)+1.51993081e-01, (float)+5.25287278e-02, (float)+9.03460830e-02, (float)-3.13828707e-01, (float)+3.09303608e-02, (float)+1.75513715e-01, (float)-2.39157781e-01, (float)+4.48320597e-01, (float)-4.59052771e-02, (float)+2.55714566e-01, (float)+2.68957704e-01, },
    };

    vector<float> layer3_biasmatrix = {
    {(float)-1.29741281e-01, (float)+1.83051378e-01, (float)-2.00942084e-02, (float)-1.45333707e-02, (float)-1.84201971e-02, (float)+7.01147616e-02, (float)+1.64620280e-01, (float)-1.95484012e-01, (float)+5.53222522e-02, (float)-2.95549445e-03, (float)+5.81931435e-02, (float)+1.61381271e-02, (float)-1.25433326e-01, (float)+2.22692937e-02, (float)-1.73236325e-01, (float)+7.32953623e-02, (float)+3.97919789e-02, (float)+4.14367914e-02, (float)-1.88474551e-01, (float)-1.11866340e-01, },
    };
    vector<vector<float>> layel3_weightmatrix = {
    {(float)-3.79723907e-01, (float)+3.96164274e-03, (float)+2.19080016e-01, (float)+1.29671037e-01, (float)+3.00807655e-01, (float)+3.45243245e-01, (float)+2.69719809e-01, (float)+3.28020424e-01, (float)-4.09428775e-01, (float)+4.90753263e-01, (float)+1.25788677e+00, (float)-1.85958639e-01, (float)-7.77057931e-02, (float)-7.49324083e-01, (float)-5.33397257e-01, (float)-2.15190679e-01, (float)+1.42438203e-01, (float)+3.08468789e-01, (float)-6.59553707e-02, (float)-2.67855048e-01, },
    {(float)+6.37590140e-02, (float)-4.74667311e-01, (float)+2.06201479e-01, (float)+2.52926528e-01, (float)+3.10887814e-01, (float)+1.95319071e-01, (float)-4.44039762e-01, (float)+3.43714952e-01, (float)-8.84505436e-02, (float)+6.49282634e-02, (float)+2.97870219e-01, (float)-3.97800744e-01, (float)-7.60561302e-02, (float)-3.45780641e-01, (float)+5.70975840e-01, (float)+1.25407055e-01, (float)+5.42687671e-03, (float)+1.62630990e-01, (float)+5.17351508e-01, (float)-9.81347710e-02, },
    {(float)+2.13177413e-01, (float)-3.37858528e-01, (float)+2.15735316e-01, (float)-2.61097670e-01, (float)+3.49516273e-01, (float)+5.92990927e-02, (float)+3.55402470e-01, (float)-3.49986553e-01, (float)+2.19975352e-01, (float)+4.49926853e-01, (float)-3.00175399e-01, (float)+2.13803072e-02, (float)-5.63070238e-01, (float)+2.18620181e-01, (float)-2.84710318e-01, (float)-2.82973230e-01, (float)+7.74637088e-02, (float)-3.12986225e-01, (float)+2.06100315e-01, (float)-2.51345456e-01, },
    {(float)-1.87825233e-01, (float)-1.94162667e+00, (float)+6.72636926e-01, (float)-2.40047723e-01, (float)-2.98054181e-02, (float)-4.66425866e-02, (float)-3.85795720e-02, (float)+2.14320749e-01, (float)+4.44862753e-01, (float)-2.74504602e-01, (float)-3.39435041e-01, (float)+5.11847436e-02, (float)+2.00064674e-01, (float)-2.93995380e-01, (float)+3.24257948e-02, (float)-2.77350634e-01, (float)-4.35923673e-02, (float)+3.99372518e-01, (float)-7.85519183e-02, (float)-2.90391207e-01, },
    {(float)+1.63187176e-01, (float)-2.62216806e-01, (float)+1.42090425e-01, (float)+4.22322541e-01, (float)+1.35618091e-01, (float)-3.32250655e-01, (float)-1.31835639e-01, (float)-3.42004925e-01, (float)-6.48177341e-02, (float)-3.34962428e-01, (float)+4.00948972e-01, (float)-1.59844458e-01, (float)-3.39585543e-02, (float)-1.29395962e-01, (float)+3.04581285e-01, (float)+5.28870821e-01, (float)-1.81530062e-02, (float)+5.13339162e-01, (float)-7.53902853e-01, (float)+2.62766570e-01, },
    {(float)+1.34777613e-02, (float)-5.87000251e-01, (float)-1.26607791e-01, (float)-1.17379099e-01, (float)+2.01676730e-02, (float)-2.70709902e-01, (float)+4.08489108e-01, (float)+5.11118293e-01, (float)-4.78331372e-02, (float)-8.08172822e-01, (float)-6.31737053e-01, (float)+1.25739053e-01, (float)-1.19169332e-01, (float)+3.45693201e-01, (float)+1.69279173e-01, (float)+4.67309095e-02, (float)+2.65802424e-02, (float)+3.65281403e-01, (float)+2.91508645e-01, (float)-5.91347832e-03, },
    {(float)+6.43120334e-02, (float)-1.91634253e-01, (float)+6.95006102e-02, (float)+2.42328808e-01, (float)-4.22908783e-01, (float)-5.56746781e-01, (float)+3.47132534e-01, (float)+1.85463309e-01, (float)+3.10665011e-01, (float)+2.43836269e-01, (float)-3.35719645e-01, (float)-5.48123777e-01, (float)+8.24262202e-02, (float)-1.49488956e-01, (float)+4.20953006e-01, (float)+2.44552538e-01, (float)+2.93123871e-01, (float)+3.92796695e-01, (float)+1.25183240e-01, (float)-2.73195267e-01, },
    {(float)+2.99723119e-01, (float)-2.37503007e-01, (float)-9.58319843e-01, (float)-3.50602239e-01, (float)-1.50072098e-01, (float)-9.50147584e-02, (float)+2.84586489e-01, (float)-3.79097462e-02, (float)-2.71351039e-01, (float)-4.50066388e-01, (float)+5.59978426e-01, (float)+4.73465919e-01, (float)+3.06118548e-01, (float)-1.90738499e-01, (float)-3.17136616e-01, (float)-3.61930758e-01, (float)+1.96969092e-01, (float)-3.65558863e-01, (float)-5.97864330e-01, (float)-8.70132595e-02, },
    {(float)-8.17695931e-02, (float)-1.19363272e+00, (float)-9.86088216e-02, (float)+6.30376935e-02, (float)+4.19927090e-01, (float)+2.36455441e-01, (float)+7.26219118e-02, (float)-1.91753969e-01, (float)+3.93098414e-01, (float)+6.58992052e-01, (float)+5.90206198e-02, (float)-3.04430556e-02, (float)+2.39279047e-01, (float)-4.75658685e-01, (float)-1.77681856e-02, (float)-2.87180543e-01, (float)+7.89332166e-02, (float)+4.86344218e-01, (float)-6.51227236e-02, (float)-5.19260094e-02, },
    {(float)+4.33036894e-01, (float)+1.08951524e-01, (float)+4.38901216e-01, (float)+1.59550324e-01, (float)-4.28432047e-01, (float)+1.23888671e-01, (float)-2.73648649e-01, (float)+5.75895786e-01, (float)-4.10643935e-01, (float)+4.92548980e-02, (float)+7.85197946e-04, (float)-3.08504462e-01, (float)-1.12201683e-01, (float)-5.88795245e-02, (float)+1.54599279e-01, (float)-1.72900051e-01, (float)-2.20936034e-02, (float)-8.72321203e-02, (float)+2.98872113e-01, (float)+2.09407046e-01, },
    {(float)-8.12305585e-02, (float)-6.15684807e-01, (float)+4.05820847e-01, (float)+3.67746167e-02, (float)+2.44984180e-02, (float)+9.45408270e-02, (float)-1.44828409e-01, (float)+4.44814004e-02, (float)-6.56353980e-02, (float)+2.49774203e-01, (float)+2.75736988e-01, (float)+1.62694871e-01, (float)-2.75552124e-01, (float)-5.97275756e-02, (float)-2.73619294e-01, (float)-6.25041351e-02, (float)+1.37822390e-01, (float)+6.98270142e-01, (float)+2.77199922e-03, (float)-2.19025597e-01, },
    {(float)-1.32989259e-02, (float)-3.31165791e-01, (float)+3.12207669e-01, (float)-1.42315850e-01, (float)+2.81373169e-02, (float)+1.24861158e-01, (float)+4.48472112e-01, (float)+2.80108806e-02, (float)-1.85780555e-01, (float)-3.60037982e-01, (float)+1.14639550e-01, (float)+2.01796204e-01, (float)-4.60069805e-01, (float)+5.88550568e-02, (float)-1.23560764e-01, (float)+3.05705994e-01, (float)+8.57421011e-02, (float)+6.67704120e-02, (float)-7.03274682e-02, (float)-1.86711639e-01, },
    {(float)-1.92500055e-01, (float)-4.54324722e-01, (float)+1.16559106e-03, (float)-3.66377920e-01, (float)+3.74631397e-02, (float)+4.62122373e-02, (float)+1.80030335e-02, (float)-3.32060248e-01, (float)+1.78643186e-02, (float)-3.87065887e-01, (float)+5.46962738e-01, (float)-3.50802273e-01, (float)+1.85700566e-01, (float)+8.74815062e-02, (float)-3.00308079e-01, (float)-4.03891981e-01, (float)+3.55183363e-01, (float)+6.39684871e-02, (float)-3.51302256e-03, (float)+4.87527512e-02, },
    {(float)+2.11178660e-01, (float)+7.81356335e-01, (float)+1.95371717e-01, (float)-2.92457789e-02, (float)+3.44266564e-01, (float)-4.75725323e-01, (float)+2.25578323e-01, (float)-9.66064811e-01, (float)-5.87617934e-01, (float)+1.12922169e-01, (float)+8.04825246e-01, (float)-6.71560541e-02, (float)-1.22341566e-01, (float)+9.59814861e-02, (float)+1.86695270e-02, (float)+1.31181985e-01, (float)+4.78610903e-01, (float)-1.56087053e+00, (float)+1.39499307e-01, (float)-5.37120923e-02, },
    {(float)+2.59321451e-01, (float)+4.51591551e-01, (float)+2.94225544e-01, (float)+3.00554723e-01, (float)+1.40241802e-01, (float)-1.57307088e-02, (float)+2.62539983e-01, (float)-4.93828893e-01, (float)+2.00774312e-01, (float)-9.95841846e-02, (float)+4.46188487e-02, (float)-8.46745446e-02, (float)-3.76699358e-01, (float)-2.77775317e-01, (float)-4.61330682e-01, (float)+1.10008225e-01, (float)+2.61049896e-01, (float)+2.54051298e-01, (float)-2.79541224e-01, (float)-3.41766894e-01, },
    {(float)+4.29174334e-01, (float)+8.80538166e-01, (float)-4.18340504e-01, (float)-3.09414983e-01, (float)+1.39982700e-01, (float)-7.35227942e-01, (float)+4.24827375e-02, (float)-8.87939930e-01, (float)+9.53125674e-03, (float)+1.04777068e-02, (float)+1.13826752e-01, (float)-5.04255891e-01, (float)-1.58160627e-01, (float)+1.49222657e-01, (float)+2.12092325e-01, (float)-2.52589405e-01, (float)+1.72029629e-01, (float)-1.07210350e+00, (float)-6.43623769e-01, (float)-1.06912799e-01, },
    {(float)-4.20393080e-01, (float)-6.02223992e-01, (float)+7.48518884e-01, (float)+2.36552224e-01, (float)+1.61030471e-01, (float)-2.75629789e-01, (float)-2.46900484e-01, (float)-2.66000628e-02, (float)+9.00073573e-02, (float)+5.39284199e-02, (float)-5.20999312e-01, (float)+5.12039438e-02, (float)-4.67868656e-01, (float)+1.23174444e-01, (float)+3.64037573e-01, (float)+6.78634226e-01, (float)-8.75248238e-02, (float)-9.89105031e-02, (float)-2.95813084e-01, (float)-3.70063111e-02, },
    {(float)-3.33520174e-01, (float)-1.53133973e-01, (float)+1.85644880e-01, (float)+7.40818903e-02, (float)-5.66387810e-02, (float)+2.65059412e-01, (float)-1.50060147e-01, (float)-2.75791705e-01, (float)+1.57778710e-01, (float)+2.85600573e-01, (float)-1.39244184e-01, (float)+1.87703874e-02, (float)+2.87081033e-01, (float)+1.64276391e-01, (float)-4.53682899e-01, (float)-2.43305907e-01, (float)+1.07084885e-01, (float)+4.30874079e-01, (float)+3.96706730e-01, (float)+9.62110385e-02, },
    {(float)-9.72981900e-02, (float)-2.01660573e-01, (float)-3.72389019e-01, (float)+1.92121267e-01, (float)-1.21244125e-01, (float)+1.84645414e-01, (float)-3.06812394e-02, (float)+8.49131308e-03, (float)+2.17364848e-01, (float)+1.51952177e-01, (float)-8.40337515e-01, (float)-8.13265219e-02, (float)-2.96978354e-01, (float)-1.56352356e-01, (float)+1.73284605e-01, (float)+7.49514550e-02, (float)-3.48767281e-01, (float)+4.33002979e-01, (float)+7.46235028e-02, (float)-1.40886605e-01, },
    {(float)-1.61007896e-01, (float)-1.86089531e-01, (float)-1.59311444e-01, (float)-2.13570833e-01, (float)+3.06685686e-01, (float)-1.80946272e-02, (float)-2.92363703e-01, (float)-1.73349828e-02, (float)-3.02100152e-01, (float)-4.00231369e-02, (float)-6.46089837e-02, (float)+1.08655781e-01, (float)+1.47528306e-01, (float)+5.89469559e-02, (float)+4.53299850e-01, (float)+6.46976978e-02, (float)-2.62179673e-01, (float)-1.02586709e-02, (float)+3.83779287e-01, (float)+3.06391805e-01, },
    };



        vector<float> layer4_biasmatrix = {
    {(float)+8.85658711e-02, (float)+1.30440176e-01, (float)+1.57286957e-01, (float)+1.99166849e-01, (float)+1.05468668e-01, },
        };
       vector<vector<float>> layel4_weightmatrix = {
    {(float)-3.79796654e-01, (float)+8.55489224e-02, (float)-1.96079984e-01, (float)-2.28885815e-01, (float)-1.12484805e-01, },
    {(float)-1.18096389e-01, (float)+8.63224491e-02, (float)-8.53372395e-01, (float)+3.07089031e-01, (float)+1.02853574e-01, },
    {(float)-1.05707183e-01, (float)-2.34325752e-01, (float)-1.13106892e-03, (float)+8.53158161e-02, (float)+6.72079027e-01, },
    {(float)+1.12596355e-01, (float)-3.56557310e-01, (float)+3.11978817e-01, (float)+1.17114060e-01, (float)+2.50213057e-01, },
    {(float)-1.60263419e-01, (float)-5.44567406e-01, (float)+2.85734802e-01, (float)-6.12525344e-01, (float)+2.27493942e-01, },
    {(float)-4.50412214e-01, (float)-2.64164329e-01, (float)+1.86928585e-01, (float)+1.36689395e-01, (float)+1.53008386e-01, },
    {(float)+6.34879231e-01, (float)+3.89694095e-01, (float)+1.66173652e-01, (float)+2.43489712e-01, (float)+2.81524509e-01, },
    {(float)-3.43103595e-02, (float)-5.69826603e-01, (float)-7.38735378e-01, (float)+2.32397661e-01, (float)-5.10263860e-01, },
    {(float)+3.94492477e-01, (float)+7.97731757e-01, (float)+7.44171381e-01, (float)-1.30988345e-01, (float)-2.81250328e-01, },
    {(float)+2.86212582e-02, (float)+6.72751144e-02, (float)-2.62504548e-01, (float)-8.66988674e-02, (float)+2.82482207e-01, },
    {(float)+4.54400256e-02, (float)-9.65893626e-01, (float)+3.32427137e-02, (float)-1.09476587e-02, (float)-1.38234153e-01, },
    {(float)-4.84076351e-01, (float)+6.09449565e-01, (float)+3.42214823e-01, (float)+3.47504206e-02, (float)+1.71333700e-01, },
    {(float)-2.88878500e-01, (float)-2.12761104e-01, (float)-4.64859866e-02, (float)+6.43446445e-02, (float)-1.58643261e-01, },
    {(float)-4.44527835e-01, (float)+4.17954415e-01, (float)-2.13731289e-01, (float)-3.04228757e-02, (float)+3.54018927e-01, },
    {(float)+9.36065242e-03, (float)-2.93679476e-01, (float)-3.76324058e-01, (float)-5.97236753e-01, (float)-5.58101177e-01, },
    {(float)+1.95446417e-01, (float)+3.95456314e-01, (float)+7.95307234e-02, (float)-6.61153719e-02, (float)+4.82656837e-01, },
    {(float)+4.69264299e-01, (float)+1.91919416e-01, (float)+5.34672379e-01, (float)-2.63899922e-01, (float)+2.46587712e-02, },
    {(float)-4.11558188e-02, (float)+1.89842656e-02, (float)+3.95699382e-01, (float)+7.71322668e-01, (float)+1.95338279e-01, },
    {(float)-1.54517069e-01, (float)+5.75042702e-02, (float)-5.29557347e-01, (float)-5.75611234e-01, (float)+1.99836027e-03, },
    {(float)-1.52133167e-01, (float)+2.33514849e-02, (float)-8.70033056e-02, (float)-6.04346156e-01, (float)-1.28341511e-01, },
        };




        vector<vector<float>> layer1_output = training_forward_neurallayer_output(input_data, layel1_weightmatrix, layer1_biasmatrix, node, function_type);

        // 第二层
    vector<vector<float>> layer2_output = training_forward_neurallayer_output(layer1_output, layel2_weightmatrix, layer2_biasmatrix, laye2_neural_nodeCount, function_type);


    vector<vector<float>> layer3_output = training_forward_neurallayer_output(layer2_output, layel3_weightmatrix, layer3_biasmatrix, laye2_neural_nodeCount, function_type);



        // 输出层
    vector<vector<float>>   output = training_forward_neurallayer_output(layer3_output, layel3_weightmatrix, layer3_biasmatrix, input_data.size(), function_type);

    vector<vector<float>>   error = calculateerror(output, input_data);
     float    mse = calculateMSE(error);

        cout <<"MSE:" << mse << endl;


    cout << "最后一次迭代输出结果：" << endl;
    print2DArray(output);

    return 0;
}*/



int main()
{

    const vector<vector<float>> initial_data = {
    {(float)+6.92191844e-01, (float)+9.95272036e-01, (float)+9.86803804e-01, (float)+8.32296560e-01, (float)+6.31590248e-01, },
    {(float)+6.91496756e-01, (float)+9.95272036e-01, (float)+9.86027557e-01, (float)+8.32042656e-01, (float)+6.30731369e-01, },
    {(float)+6.90801668e-01, (float)+9.95272036e-01, (float)+9.85251310e-01, (float)+8.31661800e-01, (float)+6.29872489e-01, },
    {(float)+6.90106580e-01, (float)+9.95295795e-01, (float)+9.84475063e-01, (float)+8.31407896e-01, (float)+6.28997093e-01, },
    {(float)+6.89411492e-01, (float)+9.95295795e-01, (float)+9.83698816e-01, (float)+8.31027041e-01, (float)+6.28138214e-01, },
    {(float)+6.88716404e-01, (float)+9.95295795e-01, (float)+9.82728508e-01, (float)+8.30773137e-01, (float)+6.27279334e-01, },
    {(float)+6.88021316e-01, (float)+9.95319553e-01, (float)+9.81952261e-01, (float)+8.30519233e-01, (float)+6.26420455e-01, },
    {(float)+6.87326228e-01, (float)+9.95319553e-01, (float)+9.81176014e-01, (float)+8.30138378e-01, (float)+6.25561575e-01, },
    {(float)+6.86631140e-01, (float)+9.95319553e-01, (float)+9.80399767e-01, (float)+8.29884474e-01, (float)+6.24702696e-01, },
    {(float)+6.85936052e-01, (float)+9.95343312e-01, (float)+9.79623520e-01, (float)+8.29630570e-01, (float)+6.23843816e-01, },
    {(float)+6.85240964e-01, (float)+9.95343312e-01, (float)+9.78653212e-01, (float)+8.29249714e-01, (float)+6.22968420e-01, },
    {(float)+6.84545876e-01, (float)+9.95343312e-01, (float)+9.77876965e-01, (float)+8.28995811e-01, (float)+6.22109540e-01, },
    {(float)+6.83850788e-01, (float)+9.95367071e-01, (float)+9.77100718e-01, (float)+8.28614955e-01, (float)+6.21250661e-01, },
    {(float)+6.83155700e-01, (float)+9.95367071e-01, (float)+9.76324471e-01, (float)+8.28361051e-01, (float)+6.20391781e-01, },
    {(float)+6.82460612e-01, (float)+9.95367071e-01, (float)+9.75548224e-01, (float)+8.28107147e-01, (float)+6.19532902e-01, },
    {(float)+6.81765524e-01, (float)+9.95390829e-01, (float)+9.74577916e-01, (float)+8.27726292e-01, (float)+6.18674022e-01, },
    {(float)+6.81070436e-01, (float)+9.95390829e-01, (float)+9.73801669e-01, (float)+8.27472388e-01, (float)+6.17815143e-01, },
    {(float)+6.80375348e-01, (float)+9.95390829e-01, (float)+9.73025422e-01, (float)+8.27091532e-01, (float)+6.16939746e-01, },
    {(float)+6.79680259e-01, (float)+9.95414588e-01, (float)+9.72249175e-01, (float)+8.26837629e-01, (float)+6.16080867e-01, },
    {(float)+6.78985171e-01, (float)+9.95414588e-01, (float)+9.71472928e-01, (float)+8.26583725e-01, (float)+6.15221987e-01, },
    {(float)+6.78290083e-01, (float)+9.95414588e-01, (float)+9.70502620e-01, (float)+8.26202869e-01, (float)+6.14363108e-01, },
    {(float)+6.77594995e-01, (float)+9.95438346e-01, (float)+9.69726373e-01, (float)+8.25948965e-01, (float)+6.13504228e-01, },
    {(float)+6.76899907e-01, (float)+9.95438346e-01, (float)+9.68950126e-01, (float)+8.25695062e-01, (float)+6.12645349e-01, },
    {(float)+6.76204819e-01, (float)+9.95438346e-01, (float)+9.68173879e-01, (float)+8.25314206e-01, (float)+6.11786469e-01, },
    {(float)+6.75509731e-01, (float)+9.95462105e-01, (float)+9.67397632e-01, (float)+8.25060302e-01, (float)+6.10927590e-01, },
    {(float)+6.74814643e-01, (float)+9.95462105e-01, (float)+9.66427324e-01, (float)+8.24679446e-01, (float)+6.10052193e-01, },
    {(float)+6.74119555e-01, (float)+9.95462105e-01, (float)+9.65651077e-01, (float)+8.24425543e-01, (float)+6.09193314e-01, },
    {(float)+6.73424467e-01, (float)+9.95485864e-01, (float)+9.64874830e-01, (float)+8.24171639e-01, (float)+6.08334434e-01, },
    {(float)+6.72729379e-01, (float)+9.95485864e-01, (float)+9.64098583e-01, (float)+8.23790783e-01, (float)+6.07475555e-01, },
    {(float)+6.72034291e-01, (float)+9.95485864e-01, (float)+9.63322337e-01, (float)+8.23536880e-01, (float)+6.06616675e-01, },
    {(float)+6.71223355e-01, (float)+9.95509622e-01, (float)+9.62546090e-01, (float)+8.23156024e-01, (float)+6.05757796e-01, },
    {(float)+6.70528267e-01, (float)+9.95509622e-01, (float)+9.61575781e-01, (float)+8.22902120e-01, (float)+6.04898916e-01, },
    {(float)+6.69833179e-01, (float)+9.95509622e-01, (float)+9.60799534e-01, (float)+8.22648216e-01, (float)+6.04023520e-01, },
    {(float)+6.69138091e-01, (float)+9.95533381e-01, (float)+9.60023287e-01, (float)+8.22267361e-01, (float)+6.03164641e-01, },
    {(float)+6.68443003e-01, (float)+9.95533381e-01, (float)+9.59247041e-01, (float)+8.22013457e-01, (float)+6.02305761e-01, },
    {(float)+6.67747915e-01, (float)+9.95557139e-01, (float)+9.58470794e-01, (float)+8.21759553e-01, (float)+6.01446882e-01, },
    {(float)+6.67052827e-01, (float)+9.95557139e-01, (float)+9.57500485e-01, (float)+8.21378697e-01, (float)+6.00588002e-01, },
    {(float)+6.66357739e-01, (float)+9.95557139e-01, (float)+9.56724238e-01, (float)+8.21124794e-01, (float)+5.99729123e-01, },
    {(float)+6.65662651e-01, (float)+9.95580898e-01, (float)+9.55947991e-01, (float)+8.20743938e-01, (float)+5.98870243e-01, },
    {(float)+6.64967563e-01, (float)+9.95580898e-01, (float)+9.55171745e-01, (float)+8.20490034e-01, (float)+5.97994847e-01, },
    {(float)+6.64272475e-01, (float)+9.95580898e-01, (float)+9.54395498e-01, (float)+8.20236131e-01, (float)+5.97135967e-01, },
    {(float)+6.63577386e-01, (float)+9.95604657e-01, (float)+9.53425189e-01, (float)+8.19855275e-01, (float)+5.96277088e-01, },
    {(float)+6.62882298e-01, (float)+9.95604657e-01, (float)+9.52648942e-01, (float)+8.19601371e-01, (float)+5.95418208e-01, },
    {(float)+6.62187210e-01, (float)+9.95604657e-01, (float)+9.51872696e-01, (float)+8.19347467e-01, (float)+5.94559329e-01, },
    {(float)+6.61492122e-01, (float)+9.95628415e-01, (float)+9.51096449e-01, (float)+8.18966612e-01, (float)+5.93700449e-01, },
    {(float)+6.60797034e-01, (float)+9.95628415e-01, (float)+9.50320202e-01, (float)+8.18712708e-01, (float)+5.92841570e-01, },
    {(float)+6.60101946e-01, (float)+9.95628415e-01, (float)+9.49349893e-01, (float)+8.18331852e-01, (float)+5.91966173e-01, },
    {(float)+6.59406858e-01, (float)+9.95652174e-01, (float)+9.48573646e-01, (float)+8.18077948e-01, (float)+5.91107294e-01, },
    {(float)+6.58711770e-01, (float)+9.95652174e-01, (float)+9.47797400e-01, (float)+8.17824045e-01, (float)+5.90248414e-01, },
    {(float)+6.58016682e-01, (float)+9.95652174e-01, (float)+9.47021153e-01, (float)+8.17443189e-01, (float)+5.89389535e-01, },
    {(float)+6.57321594e-01, (float)+9.95675933e-01, (float)+9.46244906e-01, (float)+8.17189285e-01, (float)+5.88530655e-01, },
    {(float)+6.56626506e-01, (float)+9.95675933e-01, (float)+9.45274597e-01, (float)+8.16808430e-01, (float)+5.87671776e-01, },
    {(float)+6.55931418e-01, (float)+9.95675933e-01, (float)+9.44498350e-01, (float)+8.16554526e-01, (float)+5.86812896e-01, },
    {(float)+6.55236330e-01, (float)+9.95699691e-01, (float)+9.43722104e-01, (float)+8.16300622e-01, (float)+5.85937500e-01, },
    {(float)+6.54541242e-01, (float)+9.95699691e-01, (float)+9.42945857e-01, (float)+8.15919766e-01, (float)+5.85078621e-01, },
    {(float)+6.53846154e-01, (float)+9.95699691e-01, (float)+9.42169610e-01, (float)+8.15665863e-01, (float)+5.84219741e-01, },
    {(float)+6.53151066e-01, (float)+9.95723450e-01, (float)+9.41199301e-01, (float)+8.15411959e-01, (float)+5.83360862e-01, },
    {(float)+6.52455978e-01, (float)+9.95723450e-01, (float)+9.40423055e-01, (float)+8.15031103e-01, (float)+5.82501982e-01, },
    {(float)+6.51760890e-01, (float)+9.95723450e-01, (float)+9.39646808e-01, (float)+8.14777199e-01, (float)+5.81643103e-01, },
    {(float)+6.51065802e-01, (float)+9.95747208e-01, (float)+9.38870561e-01, (float)+8.14396344e-01, (float)+5.80784223e-01, },
    {(float)+6.50370714e-01, (float)+9.95747208e-01, (float)+9.38094314e-01, (float)+8.14142440e-01, (float)+5.79908827e-01, },
    {(float)+6.49675626e-01, (float)+9.95747208e-01, (float)+9.37124005e-01, (float)+8.13888536e-01, (float)+5.79049947e-01, },
    {(float)+6.48980538e-01, (float)+9.95770967e-01, (float)+9.36347759e-01, (float)+8.13507681e-01, (float)+5.78191068e-01, },
    {(float)+6.48285449e-01, (float)+9.95770967e-01, (float)+9.35571512e-01, (float)+8.13253777e-01, (float)+5.77332188e-01, },
    {(float)+6.47590361e-01, (float)+9.95770967e-01, (float)+9.34795265e-01, (float)+8.12872921e-01, (float)+5.76473309e-01, },
    {(float)+6.46895273e-01, (float)+9.95794726e-01, (float)+9.34019018e-01, (float)+8.12619017e-01, (float)+5.75614429e-01, },
    {(float)+6.46200185e-01, (float)+9.95794726e-01, (float)+9.33048709e-01, (float)+8.12365114e-01, (float)+5.74755550e-01, },
    {(float)+6.45505097e-01, (float)+9.95794726e-01, (float)+9.32272463e-01, (float)+8.11984258e-01, (float)+5.73880153e-01, },
    {(float)+6.44810009e-01, (float)+9.95818484e-01, (float)+9.31496216e-01, (float)+8.11730354e-01, (float)+5.73021274e-01, },
    {(float)+6.44114921e-01, (float)+9.95818484e-01, (float)+9.30719969e-01, (float)+8.11476450e-01, (float)+5.72162394e-01, },
    {(float)+6.43419833e-01, (float)+9.95818484e-01, (float)+9.29943722e-01, (float)+8.11095595e-01, (float)+5.71303515e-01, },
    {(float)+6.42724745e-01, (float)+9.95842243e-01, (float)+9.28973414e-01, (float)+8.10841691e-01, (float)+5.70444635e-01, },
    {(float)+6.42029657e-01, (float)+9.95842243e-01, (float)+9.28197167e-01, (float)+8.10460835e-01, (float)+5.69585756e-01, },
    {(float)+6.41334569e-01, (float)+9.95842243e-01, (float)+9.27420920e-01, (float)+8.10206932e-01, (float)+5.68726876e-01, },
    {(float)+6.40639481e-01, (float)+9.95866001e-01, (float)+9.26644673e-01, (float)+8.09953028e-01, (float)+5.67867997e-01, },
    {(float)+6.39944393e-01, (float)+9.95866001e-01, (float)+9.25868426e-01, (float)+8.09572172e-01, (float)+5.66992600e-01, },
    {(float)+6.39249305e-01, (float)+9.95866001e-01, (float)+9.24898118e-01, (float)+8.09318268e-01, (float)+5.66133721e-01, },
    {(float)+6.38554217e-01, (float)+9.95889760e-01, (float)+9.24121871e-01, (float)+8.09064365e-01, (float)+5.65274841e-01, },
    {(float)+6.37859129e-01, (float)+9.95889760e-01, (float)+9.23345624e-01, (float)+8.08683509e-01, (float)+5.64415962e-01, },
    {(float)+6.37164041e-01, (float)+9.95889760e-01, (float)+9.22569377e-01, (float)+8.08429605e-01, (float)+5.63557082e-01, },
    {(float)+6.36468953e-01, (float)+9.95913519e-01, (float)+9.21793130e-01, (float)+8.08048750e-01, (float)+5.62698203e-01, },
    {(float)+6.35773865e-01, (float)+9.95913519e-01, (float)+9.20822822e-01, (float)+8.07794846e-01, (float)+5.61839323e-01, },
    {(float)+6.35078777e-01, (float)+9.95913519e-01, (float)+9.20046575e-01, (float)+8.07540942e-01, (float)+5.60963927e-01, },
    {(float)+6.34383689e-01, (float)+9.95937277e-01, (float)+9.19270328e-01, (float)+8.07160086e-01, (float)+5.60105048e-01, },
    {(float)+6.33688601e-01, (float)+9.95937277e-01, (float)+9.18494081e-01, (float)+8.06906183e-01, (float)+5.59246168e-01, },
    {(float)+6.32993513e-01, (float)+9.95937277e-01, (float)+9.17717834e-01, (float)+8.06525327e-01, (float)+5.58387289e-01, },
    {(float)+6.32298424e-01, (float)+9.95961036e-01, (float)+9.16747526e-01, (float)+8.06271423e-01, (float)+5.57528409e-01, },
    {(float)+6.31603336e-01, (float)+9.95961036e-01, (float)+9.15971279e-01, (float)+8.06017519e-01, (float)+5.56669530e-01, },
    {(float)+6.30908248e-01, (float)+9.95961036e-01, (float)+9.15195032e-01, (float)+8.05636664e-01, (float)+5.55810650e-01, },
    {(float)+6.30213160e-01, (float)+9.95984794e-01, (float)+9.14418785e-01, (float)+8.05382760e-01, (float)+5.54935254e-01, },
    {(float)+6.29518072e-01, (float)+9.95984794e-01, (float)+9.13642538e-01, (float)+8.05128856e-01, (float)+5.54076374e-01, },
    {(float)+6.28822984e-01, (float)+9.95984794e-01, (float)+9.12672230e-01, (float)+8.04748001e-01, (float)+5.53217495e-01, },
    {(float)+6.28127896e-01, (float)+9.96008553e-01, (float)+9.11895983e-01, (float)+8.04494097e-01, (float)+5.52358615e-01, },
    {(float)+6.27432808e-01, (float)+9.96008553e-01, (float)+9.11119736e-01, (float)+8.04113241e-01, (float)+5.51499736e-01, },
    {(float)+6.26737720e-01, (float)+9.96008553e-01, (float)+9.10343489e-01, (float)+8.03859337e-01, (float)+5.50640856e-01, },
    {(float)+6.26042632e-01, (float)+9.96032312e-01, (float)+9.09567242e-01, (float)+8.03605434e-01, (float)+5.49781977e-01, },
    {(float)+6.25347544e-01, (float)+9.96032312e-01, (float)+9.08790996e-01, (float)+8.03224578e-01, (float)+5.48906580e-01, },
    {(float)+6.24652456e-01, (float)+9.96032312e-01, (float)+9.07820687e-01, (float)+8.02970674e-01, (float)+5.48047701e-01, },
    {(float)+6.23957368e-01, (float)+9.96056070e-01, (float)+9.07044440e-01, (float)+8.02589818e-01, (float)+5.47188821e-01, },
    {(float)+6.23262280e-01, (float)+9.96056070e-01, (float)+9.06268193e-01, (float)+8.02335915e-01, (float)+5.46329942e-01, },
    {(float)+6.22567192e-01, (float)+9.96056070e-01, (float)+9.05491946e-01, (float)+8.02082011e-01, (float)+5.45471062e-01, },
    {(float)+6.21872104e-01, (float)+9.96079829e-01, (float)+9.04715700e-01, (float)+8.01701155e-01, (float)+5.44612183e-01, },
    {(float)+6.21177016e-01, (float)+9.96079829e-01, (float)+9.03745391e-01, (float)+8.01447251e-01, (float)+5.43753303e-01, },
    {(float)+6.20481928e-01, (float)+9.96079829e-01, (float)+9.02969144e-01, (float)+8.01193348e-01, (float)+5.42877907e-01, },
    {(float)+6.19786840e-01, (float)+9.96103588e-01, (float)+9.02192897e-01, (float)+8.00812492e-01, (float)+5.42019027e-01, },
    {(float)+6.18975904e-01, (float)+9.96103588e-01, (float)+9.01416650e-01, (float)+8.00558588e-01, (float)+5.41160148e-01, },
    {(float)+6.18280816e-01, (float)+9.96103588e-01, (float)+9.00640404e-01, (float)+8.00177733e-01, (float)+5.40301268e-01, },
    {(float)+6.17585728e-01, (float)+9.96127346e-01, (float)+8.99670095e-01, (float)+7.99923829e-01, (float)+5.39442389e-01, },
    {(float)+6.16890639e-01, (float)+9.96127346e-01, (float)+8.98893848e-01, (float)+7.99669925e-01, (float)+5.38583510e-01, },
    {(float)+6.16195551e-01, (float)+9.96127346e-01, (float)+8.98117601e-01, (float)+7.99289069e-01, (float)+5.37724630e-01, },
    {(float)+6.15500463e-01, (float)+9.96151105e-01, (float)+8.97341355e-01, (float)+7.99035166e-01, (float)+5.36849234e-01, },
    {(float)+6.14805375e-01, (float)+9.96151105e-01, (float)+8.96565108e-01, (float)+7.98781262e-01, (float)+5.35990354e-01, },
    {(float)+6.14110287e-01, (float)+9.96151105e-01, (float)+8.95594799e-01, (float)+7.98400406e-01, (float)+5.35131475e-01, },
    {(float)+6.13415199e-01, (float)+9.96174863e-01, (float)+8.94818552e-01, (float)+7.98146502e-01, (float)+5.34272595e-01, },
    {(float)+6.12720111e-01, (float)+9.96174863e-01, (float)+8.94042305e-01, (float)+7.97765647e-01, (float)+5.33413716e-01, },
    {(float)+6.12025023e-01, (float)+9.96174863e-01, (float)+8.93266059e-01, (float)+7.97511743e-01, (float)+5.32554836e-01, },
    {(float)+6.11329935e-01, (float)+9.96198622e-01, (float)+8.92489812e-01, (float)+7.97257839e-01, (float)+5.31695957e-01, },
    {(float)+6.10634847e-01, (float)+9.96198622e-01, (float)+8.91519503e-01, (float)+7.96876984e-01, (float)+5.30820560e-01, },
    {(float)+6.09939759e-01, (float)+9.96198622e-01, (float)+8.90743256e-01, (float)+7.96623080e-01, (float)+5.29961681e-01, },
    {(float)+6.09244671e-01, (float)+9.96222381e-01, (float)+8.89967010e-01, (float)+7.96242224e-01, (float)+5.29102801e-01, },
    {(float)+6.08549583e-01, (float)+9.96222381e-01, (float)+8.89190763e-01, (float)+7.95988320e-01, (float)+5.28243922e-01, },
    {(float)+6.07854495e-01, (float)+9.96246139e-01, (float)+8.88414516e-01, (float)+7.95734417e-01, (float)+5.27385042e-01, },
    {(float)+6.07159407e-01, (float)+9.96246139e-01, (float)+8.87444207e-01, (float)+7.95353561e-01, (float)+5.26526163e-01, },
    {(float)+6.06464319e-01, (float)+9.96246139e-01, (float)+8.86667960e-01, (float)+7.95099657e-01, (float)+5.25667283e-01, },
    {(float)+6.05769231e-01, (float)+9.96269898e-01, (float)+8.85891714e-01, (float)+7.94845753e-01, (float)+5.24791887e-01, },
    {(float)+6.05074143e-01, (float)+9.96269898e-01, (float)+8.85115467e-01, (float)+7.94464898e-01, (float)+5.23933007e-01, },
    {(float)+6.04379055e-01, (float)+9.96269898e-01, (float)+8.84339220e-01, (float)+7.94210994e-01, (float)+5.23074128e-01, },
    {(float)+6.03683967e-01, (float)+9.96293656e-01, (float)+8.83368911e-01, (float)+7.93830138e-01, (float)+5.22215248e-01, },
    {(float)+6.02988879e-01, (float)+9.96293656e-01, (float)+8.82592664e-01, (float)+7.93576235e-01, (float)+5.21356369e-01, },
    {(float)+6.02293791e-01, (float)+9.96293656e-01, (float)+8.81816418e-01, (float)+7.93322331e-01, (float)+5.20497489e-01, },
    {(float)+6.01598703e-01, (float)+9.96317415e-01, (float)+8.81040171e-01, (float)+7.92941475e-01, (float)+5.19638610e-01, },
    {(float)+6.00903614e-01, (float)+9.96317415e-01, (float)+8.80263924e-01, (float)+7.92687571e-01, (float)+5.18779730e-01, },
    {(float)+6.00208526e-01, (float)+9.96317415e-01, (float)+8.79293615e-01, (float)+7.92306716e-01, (float)+5.17904334e-01, },
    {(float)+5.99513438e-01, (float)+9.96341174e-01, (float)+8.78517369e-01, (float)+7.92052812e-01, (float)+5.17045455e-01, },
    {(float)+5.98818350e-01, (float)+9.96341174e-01, (float)+8.77741122e-01, (float)+7.91798908e-01, (float)+5.16186575e-01, },
    {(float)+5.98123262e-01, (float)+9.96341174e-01, (float)+8.76964875e-01, (float)+7.91418053e-01, (float)+5.15327696e-01, },
    {(float)+5.97428174e-01, (float)+9.96364932e-01, (float)+8.76188628e-01, (float)+7.91164149e-01, (float)+5.14468816e-01, },
    {(float)+5.96733086e-01, (float)+9.96364932e-01, (float)+8.75218319e-01, (float)+7.90910245e-01, (float)+5.13609937e-01, },
    {(float)+5.96037998e-01, (float)+9.96364932e-01, (float)+8.74442073e-01, (float)+7.90529389e-01, (float)+5.12751057e-01, },
    {(float)+5.95342910e-01, (float)+9.96388691e-01, (float)+8.73665826e-01, (float)+7.90275486e-01, (float)+5.11875661e-01, },
    {(float)+5.94647822e-01, (float)+9.96388691e-01, (float)+8.72889579e-01, (float)+7.89894630e-01, (float)+5.11016781e-01, },
    {(float)+5.93952734e-01, (float)+9.96388691e-01, (float)+8.72113332e-01, (float)+7.89640726e-01, (float)+5.10157902e-01, },
    {(float)+5.93257646e-01, (float)+9.96412450e-01, (float)+8.71143023e-01, (float)+7.89386822e-01, (float)+5.09299022e-01, },
    {(float)+5.92562558e-01, (float)+9.96412450e-01, (float)+8.70366777e-01, (float)+7.89005967e-01, (float)+5.08440143e-01, },
    {(float)+5.91867470e-01, (float)+9.96412450e-01, (float)+8.69590530e-01, (float)+7.88752063e-01, (float)+5.07581263e-01, },
    {(float)+5.91172382e-01, (float)+9.96436208e-01, (float)+8.68814283e-01, (float)+7.88498159e-01, (float)+5.06722384e-01, },
    {(float)+5.90477294e-01, (float)+9.96436208e-01, (float)+8.68038036e-01, (float)+7.88117304e-01, (float)+5.05846987e-01, },
    {(float)+5.89782206e-01, (float)+9.96436208e-01, (float)+8.67067728e-01, (float)+7.87863400e-01, (float)+5.04988108e-01, },
    {(float)+5.89087118e-01, (float)+9.96459967e-01, (float)+8.66291481e-01, (float)+7.87482544e-01, (float)+5.04129228e-01, },
    {(float)+5.88392030e-01, (float)+9.96459967e-01, (float)+8.65515234e-01, (float)+7.87228640e-01, (float)+5.03270349e-01, },
    {(float)+5.87696942e-01, (float)+9.96459967e-01, (float)+8.64738987e-01, (float)+7.86974737e-01, (float)+5.02411469e-01, },
    {(float)+5.87001854e-01, (float)+9.96483725e-01, (float)+8.63962740e-01, (float)+7.86593881e-01, (float)+5.01552590e-01, },
    {(float)+5.86306766e-01, (float)+9.96483725e-01, (float)+8.62992432e-01, (float)+7.86339977e-01, (float)+5.00693710e-01, },
    {(float)+5.85611677e-01, (float)+9.96483725e-01, (float)+8.62216185e-01, (float)+7.85959121e-01, (float)+4.99818314e-01, },
    {(float)+5.84916589e-01, (float)+9.96507484e-01, (float)+8.61439938e-01, (float)+7.85705218e-01, (float)+4.98959434e-01, },
    {(float)+5.84221501e-01, (float)+9.96507484e-01, (float)+8.60663691e-01, (float)+7.85451314e-01, (float)+4.98100555e-01, },
    {(float)+5.83526413e-01, (float)+9.96507484e-01, (float)+8.59887444e-01, (float)+7.85070458e-01, (float)+4.97241675e-01, },
    {(float)+5.82831325e-01, (float)+9.96531243e-01, (float)+8.58917136e-01, (float)+7.84816555e-01, (float)+4.96382796e-01, },
    {(float)+5.82136237e-01, (float)+9.96531243e-01, (float)+8.58140889e-01, (float)+7.84562651e-01, (float)+4.95523916e-01, },
    {(float)+5.81441149e-01, (float)+9.96531243e-01, (float)+8.57364642e-01, (float)+7.84181795e-01, (float)+4.94665037e-01, },
    {(float)+5.80746061e-01, (float)+9.96555001e-01, (float)+8.56588395e-01, (float)+7.83927891e-01, (float)+4.93789641e-01, },
    {(float)+5.80050973e-01, (float)+9.96555001e-01, (float)+8.55812148e-01, (float)+7.83547036e-01, (float)+4.92930761e-01, },
    {(float)+5.79355885e-01, (float)+9.96555001e-01, (float)+8.55035901e-01, (float)+7.83293132e-01, (float)+4.92071882e-01, },
    {(float)+5.78660797e-01, (float)+9.96578760e-01, (float)+8.54065593e-01, (float)+7.83039228e-01, (float)+4.91213002e-01, },
    {(float)+5.77965709e-01, (float)+9.96578760e-01, (float)+8.53289346e-01, (float)+7.82658372e-01, (float)+4.90354123e-01, },
    {(float)+5.77270621e-01, (float)+9.96578760e-01, (float)+8.52513099e-01, (float)+7.82404469e-01, (float)+4.89495243e-01, },
    {(float)+5.76575533e-01, (float)+9.96602518e-01, (float)+8.51736852e-01, (float)+7.82023613e-01, (float)+4.88636364e-01, },
    {(float)+5.75880445e-01, (float)+9.96602518e-01, (float)+8.50960605e-01, (float)+7.81769709e-01, (float)+4.87760967e-01, },
    {(float)+5.75185357e-01, (float)+9.96602518e-01, (float)+8.49990297e-01, (float)+7.81515806e-01, (float)+4.86902088e-01, },
    {(float)+5.74490269e-01, (float)+9.96626277e-01, (float)+8.49214050e-01, (float)+7.81134950e-01, (float)+4.86043208e-01, },
    {(float)+5.73795181e-01, (float)+9.96626277e-01, (float)+8.48437803e-01, (float)+7.80881046e-01, (float)+4.85184329e-01, },
    {(float)+5.73100093e-01, (float)+9.96626277e-01, (float)+8.47661556e-01, (float)+7.80627142e-01, (float)+4.84325449e-01, },
    {(float)+5.72405005e-01, (float)+9.96650036e-01, (float)+8.46885310e-01, (float)+7.80246287e-01, (float)+4.83466570e-01, },
    {(float)+5.71709917e-01, (float)+9.96650036e-01, (float)+8.45915001e-01, (float)+7.79992383e-01, (float)+4.82607690e-01, },
    {(float)+5.71014829e-01, (float)+9.96650036e-01, (float)+8.45138754e-01, (float)+7.79611527e-01, (float)+4.81732294e-01, },
    {(float)+5.70319741e-01, (float)+9.96673794e-01, (float)+8.44362507e-01, (float)+7.79357623e-01, (float)+4.80873414e-01, },
    {(float)+5.69624652e-01, (float)+9.96673794e-01, (float)+8.43586260e-01, (float)+7.79103720e-01, (float)+4.80014535e-01, },
    {(float)+5.68929564e-01, (float)+9.96673794e-01, (float)+8.42810014e-01, (float)+7.78722864e-01, (float)+4.79155655e-01, },
    {(float)+5.68234476e-01, (float)+9.96697553e-01, (float)+8.41839705e-01, (float)+7.78468960e-01, (float)+4.78296776e-01, },
    {(float)+5.67539388e-01, (float)+9.96697553e-01, (float)+8.41063458e-01, (float)+7.78215056e-01, (float)+4.77437896e-01, },
    {(float)+5.66728452e-01, (float)+9.96697553e-01, (float)+8.40287211e-01, (float)+7.77834201e-01, (float)+4.76579017e-01, },
    {(float)+5.66033364e-01, (float)+9.96721311e-01, (float)+8.39510964e-01, (float)+7.77580297e-01, (float)+4.75703621e-01, },
    {(float)+5.65338276e-01, (float)+9.96721311e-01, (float)+8.38734718e-01, (float)+7.77199441e-01, (float)+4.74844741e-01, },
    {(float)+5.64643188e-01, (float)+9.96721311e-01, (float)+8.37764409e-01, (float)+7.76945538e-01, (float)+4.73985862e-01, },
    {(float)+5.63948100e-01, (float)+9.96745070e-01, (float)+8.36988162e-01, (float)+7.76691634e-01, (float)+4.73126982e-01, },
    {(float)+5.63253012e-01, (float)+9.96745070e-01, (float)+8.36211915e-01, (float)+7.76310778e-01, (float)+4.72268103e-01, },
    {(float)+5.62557924e-01, (float)+9.96745070e-01, (float)+8.35435669e-01, (float)+7.76056874e-01, (float)+4.71409223e-01, },
    {(float)+5.61862836e-01, (float)+9.96768829e-01, (float)+8.34659422e-01, (float)+7.75676019e-01, (float)+4.70550344e-01, },
    {(float)+5.61167748e-01, (float)+9.96768829e-01, (float)+8.33689113e-01, (float)+7.75422115e-01, (float)+4.69691464e-01, },
    {(float)+5.60472660e-01, (float)+9.96768829e-01, (float)+8.32912866e-01, (float)+7.75168211e-01, (float)+4.68816068e-01, },
    {(float)+5.59777572e-01, (float)+9.96792587e-01, (float)+8.32136619e-01, (float)+7.74787356e-01, (float)+4.67957188e-01, },
    {(float)+5.59082484e-01, (float)+9.96792587e-01, (float)+8.31360373e-01, (float)+7.74533452e-01, (float)+4.67098309e-01, },
    {(float)+5.58387396e-01, (float)+9.96792587e-01, (float)+8.30584126e-01, (float)+7.74279548e-01, (float)+4.66239429e-01, },
    {(float)+5.57692308e-01, (float)+9.96816346e-01, (float)+8.29613817e-01, (float)+7.73898692e-01, (float)+4.65380550e-01, },
    {(float)+5.56997220e-01, (float)+9.96816346e-01, (float)+8.28837570e-01, (float)+7.73644789e-01, (float)+4.64521670e-01, },
    {(float)+5.56302132e-01, (float)+9.96816346e-01, (float)+8.28061324e-01, (float)+7.73263933e-01, (float)+4.63662791e-01, },
    {(float)+5.55607044e-01, (float)+9.96840105e-01, (float)+8.27285077e-01, (float)+7.73010029e-01, (float)+4.62787394e-01, },
    {(float)+5.54911956e-01, (float)+9.96840105e-01, (float)+8.26508830e-01, (float)+7.72756125e-01, (float)+4.61928515e-01, },
    {(float)+5.54216867e-01, (float)+9.96840105e-01, (float)+8.25538521e-01, (float)+7.72375270e-01, (float)+4.61069635e-01, },
    {(float)+5.53521779e-01, (float)+9.96863863e-01, (float)+8.24762274e-01, (float)+7.72121366e-01, (float)+4.60210756e-01, },
    {(float)+5.52826691e-01, (float)+9.96863863e-01, (float)+8.23986028e-01, (float)+7.71740510e-01, (float)+4.59351876e-01, },
    {(float)+5.52131603e-01, (float)+9.96863863e-01, (float)+8.23209781e-01, (float)+7.71486607e-01, (float)+4.58492997e-01, },
    {(float)+5.51436515e-01, (float)+9.96887622e-01, (float)+8.22433534e-01, (float)+7.71232703e-01, (float)+4.57634117e-01, },
    {(float)+5.50741427e-01, (float)+9.96887622e-01, (float)+8.21463225e-01, (float)+7.70851847e-01, (float)+4.56758721e-01, },
    {(float)+5.50046339e-01, (float)+9.96887622e-01, (float)+8.20686978e-01, (float)+7.70597943e-01, (float)+4.55899841e-01, },
    {(float)+5.49351251e-01, (float)+9.96911380e-01, (float)+8.19910732e-01, (float)+7.70344040e-01, (float)+4.55040962e-01, },
    {(float)+5.48656163e-01, (float)+9.96911380e-01, (float)+8.19134485e-01, (float)+7.69963184e-01, (float)+4.54182082e-01, },
    {(float)+5.47961075e-01, (float)+9.96935139e-01, (float)+8.18358238e-01, (float)+7.69709280e-01, (float)+4.53323203e-01, },
    {(float)+5.47265987e-01, (float)+9.96935139e-01, (float)+8.17387929e-01, (float)+7.69328425e-01, (float)+4.52464323e-01, },
    {(float)+5.46570899e-01, (float)+9.96935139e-01, (float)+8.16611683e-01, (float)+7.69074521e-01, (float)+4.51605444e-01, },
    {(float)+5.45875811e-01, (float)+9.96958898e-01, (float)+8.15835436e-01, (float)+7.68820617e-01, (float)+4.50730048e-01, },
    {(float)+5.45180723e-01, (float)+9.96958898e-01, (float)+8.15059189e-01, (float)+7.68439761e-01, (float)+4.49871168e-01, },
    {(float)+5.44485635e-01, (float)+9.96958898e-01, (float)+8.14282942e-01, (float)+7.68185858e-01, (float)+4.49012289e-01, },
    {(float)+5.43790547e-01, (float)+9.96982656e-01, (float)+8.13312633e-01, (float)+7.67931954e-01, (float)+4.48153409e-01, },
    {(float)+5.43095459e-01, (float)+9.96982656e-01, (float)+8.12536387e-01, (float)+7.67551098e-01, (float)+4.47294530e-01, },
    {(float)+5.42400371e-01, (float)+9.96982656e-01, (float)+8.11760140e-01, (float)+7.67297194e-01, (float)+4.46435650e-01, },
    {(float)+5.41705283e-01, (float)+9.97006415e-01, (float)+8.10983893e-01, (float)+7.66916339e-01, (float)+4.45576771e-01, },
    {(float)+5.41010195e-01, (float)+9.97006415e-01, (float)+8.10207646e-01, (float)+7.66662435e-01, (float)+4.44701374e-01, },
    {(float)+5.40315107e-01, (float)+9.97006415e-01, (float)+8.09237337e-01, (float)+7.66408531e-01, (float)+4.43842495e-01, },
    {(float)+5.39620019e-01, (float)+9.97030173e-01, (float)+8.08461091e-01, (float)+7.66027676e-01, (float)+4.42983615e-01, },
    {(float)+5.38924930e-01, (float)+9.97030173e-01, (float)+8.07684844e-01, (float)+7.65773772e-01, (float)+4.42124736e-01, },
    {(float)+5.38229842e-01, (float)+9.97030173e-01, (float)+8.06908597e-01, (float)+7.65392916e-01, (float)+4.41265856e-01, },
    {(float)+5.37534754e-01, (float)+9.97053932e-01, (float)+8.06132350e-01, (float)+7.65139012e-01, (float)+4.40406977e-01, },
    {(float)+5.36839666e-01, (float)+9.97053932e-01, (float)+8.05162042e-01, (float)+7.64885109e-01, (float)+4.39548097e-01, },
    {(float)+5.36144578e-01, (float)+9.97053932e-01, (float)+8.04385795e-01, (float)+7.64504253e-01, (float)+4.38672701e-01, },
    {(float)+5.35449490e-01, (float)+9.97077691e-01, (float)+8.03609548e-01, (float)+7.64250349e-01, (float)+4.37813821e-01, },
    {(float)+5.34754402e-01, (float)+9.97077691e-01, (float)+8.02833301e-01, (float)+7.63996445e-01, (float)+4.36954942e-01, },
    {(float)+5.34059314e-01, (float)+9.97077691e-01, (float)+8.02057054e-01, (float)+7.63615590e-01, (float)+4.36096062e-01, },
    {(float)+5.33364226e-01, (float)+9.97101449e-01, (float)+8.01280807e-01, (float)+7.63361686e-01, (float)+4.35237183e-01, },
    {(float)+5.32669138e-01, (float)+9.97101449e-01, (float)+8.00310499e-01, (float)+7.62980830e-01, (float)+4.34378303e-01, },
    {(float)+5.31974050e-01, (float)+9.97101449e-01, (float)+7.99534252e-01, (float)+7.62726926e-01, (float)+4.33519424e-01, },
    {(float)+5.31278962e-01, (float)+9.97125208e-01, (float)+7.98758005e-01, (float)+7.62473023e-01, (float)+4.32644027e-01, },
    {(float)+5.30583874e-01, (float)+9.97125208e-01, (float)+7.97981758e-01, (float)+7.62092167e-01, (float)+4.31785148e-01, },
    {(float)+5.29888786e-01, (float)+9.97125208e-01, (float)+7.97205511e-01, (float)+7.61838263e-01, (float)+4.30926268e-01, },
    {(float)+5.29193698e-01, (float)+9.97148967e-01, (float)+7.96235203e-01, (float)+7.61457408e-01, (float)+4.30067389e-01, },
    {(float)+5.28498610e-01, (float)+9.97148967e-01, (float)+7.95458956e-01, (float)+7.61203504e-01, (float)+4.29208510e-01, },
    {(float)+5.27803522e-01, (float)+9.97148967e-01, (float)+7.94682709e-01, (float)+7.60949600e-01, (float)+4.28349630e-01, },
    {(float)+5.27108434e-01, (float)+9.97172725e-01, (float)+7.93906462e-01, (float)+7.60568744e-01, (float)+4.27490751e-01, },
    {(float)+5.26413346e-01, (float)+9.97172725e-01, (float)+7.93130215e-01, (float)+7.60314841e-01, (float)+4.26631871e-01, },
    {(float)+5.25718258e-01, (float)+9.97172725e-01, (float)+7.92159907e-01, (float)+7.60060937e-01, (float)+4.25756475e-01, },
    {(float)+5.25023170e-01, (float)+9.97196484e-01, (float)+7.91383660e-01, (float)+7.59680081e-01, (float)+4.24897595e-01, },
    {(float)+5.24328082e-01, (float)+9.97196484e-01, (float)+7.90607413e-01, (float)+7.59426177e-01, (float)+4.24038716e-01, },
    {(float)+5.23632994e-01, (float)+9.97196484e-01, (float)+7.89831166e-01, (float)+7.59045322e-01, (float)+4.23179836e-01, },
    {(float)+5.22937905e-01, (float)+9.97220242e-01, (float)+7.89054919e-01, (float)+7.58791418e-01, (float)+4.22320957e-01, },
    {(float)+5.22242817e-01, (float)+9.97220242e-01, (float)+7.88084611e-01, (float)+7.58537514e-01, (float)+4.21462077e-01, },
    {(float)+5.21547729e-01, (float)+9.97220242e-01, (float)+7.87308364e-01, (float)+7.58156659e-01, (float)+4.20603198e-01, },
    {(float)+5.20852641e-01, (float)+9.97244001e-01, (float)+7.86532117e-01, (float)+7.57902755e-01, (float)+4.19727801e-01, },
    {(float)+5.20157553e-01, (float)+9.97244001e-01, (float)+7.85755870e-01, (float)+7.57648851e-01, (float)+4.18868922e-01, },
    {(float)+5.19462465e-01, (float)+9.97244001e-01, (float)+7.84979624e-01, (float)+7.57267995e-01, (float)+4.18010042e-01, },
    {(float)+5.18767377e-01, (float)+9.97267760e-01, (float)+7.84009315e-01, (float)+7.57014092e-01, (float)+4.17151163e-01, },
    {(float)+5.18072289e-01, (float)+9.97267760e-01, (float)+7.83233068e-01, (float)+7.56633236e-01, (float)+4.16292283e-01, },
    {(float)+5.17377201e-01, (float)+9.97267760e-01, (float)+7.82456821e-01, (float)+7.56379332e-01, (float)+4.15433404e-01, },
    {(float)+5.16682113e-01, (float)+9.97291518e-01, (float)+7.81680574e-01, (float)+7.56125428e-01, (float)+4.14574524e-01, },
    {(float)+5.15987025e-01, (float)+9.97291518e-01, (float)+7.80904328e-01, (float)+7.55744573e-01, (float)+4.13699128e-01, },
    {(float)+5.15291937e-01, (float)+9.97291518e-01, (float)+7.79934019e-01, (float)+7.55490669e-01, (float)+4.12840248e-01, },
    {(float)+5.14481001e-01, (float)+9.97315277e-01, (float)+7.79157772e-01, (float)+7.55109813e-01, (float)+4.11981369e-01, },
    {(float)+5.13785913e-01, (float)+9.97315277e-01, (float)+7.78381525e-01, (float)+7.54855910e-01, (float)+4.11122489e-01, },
    {(float)+5.13090825e-01, (float)+9.97315277e-01, (float)+7.77605278e-01, (float)+7.54602006e-01, (float)+4.10263610e-01, },
    {(float)+5.12395737e-01, (float)+9.97339035e-01, (float)+7.76829032e-01, (float)+7.54221150e-01, (float)+4.09404730e-01, },
    {(float)+5.11700649e-01, (float)+9.97339035e-01, (float)+7.75858723e-01, (float)+7.53967246e-01, (float)+4.08545851e-01, },
    {(float)+5.11005561e-01, (float)+9.97339035e-01, (float)+7.75082476e-01, (float)+7.53713343e-01, (float)+4.07670455e-01, },
    {(float)+5.10310473e-01, (float)+9.97362794e-01, (float)+7.74306229e-01, (float)+7.53332487e-01, (float)+4.06811575e-01, },
    {(float)+5.09615385e-01, (float)+9.97362794e-01, (float)+7.73529983e-01, (float)+7.53078583e-01, (float)+4.05952696e-01, },
    {(float)+5.08920297e-01, (float)+9.97362794e-01, (float)+7.72753736e-01, (float)+7.52697728e-01, (float)+4.05093816e-01, },
    {(float)+5.08225209e-01, (float)+9.97386553e-01, (float)+7.71783427e-01, (float)+7.52443824e-01, (float)+4.04234937e-01, },
    {(float)+5.07530120e-01, (float)+9.97386553e-01, (float)+7.71007180e-01, (float)+7.52189920e-01, (float)+4.03376057e-01, },
    {(float)+5.06835032e-01, (float)+9.97386553e-01, (float)+7.70230933e-01, (float)+7.51809064e-01, (float)+4.02517178e-01, },
    {(float)+5.06139944e-01, (float)+9.97410311e-01, (float)+7.69454687e-01, (float)+7.51555161e-01, (float)+4.01641781e-01, },
    {(float)+5.05444856e-01, (float)+9.97410311e-01, (float)+7.68678440e-01, (float)+7.51174305e-01, (float)+4.00782902e-01, },
    {(float)+5.04749768e-01, (float)+9.97410311e-01, (float)+7.67708131e-01, (float)+7.50920401e-01, (float)+3.99924022e-01, },
    {(float)+5.04054680e-01, (float)+9.97434070e-01, (float)+7.66931884e-01, (float)+7.50666497e-01, (float)+3.99065143e-01, },
    {(float)+5.03359592e-01, (float)+9.97434070e-01, (float)+7.66155637e-01, (float)+7.50285642e-01, (float)+3.98206263e-01, },
    {(float)+5.02664504e-01, (float)+9.97434070e-01, (float)+7.65379391e-01, (float)+7.50031738e-01, (float)+3.97347384e-01, },
    {(float)+5.01969416e-01, (float)+9.97457828e-01, (float)+7.64603144e-01, (float)+7.49777834e-01, (float)+3.96488504e-01, },
    {(float)+5.01274328e-01, (float)+9.97457828e-01, (float)+7.63632835e-01, (float)+7.49396979e-01, (float)+3.95613108e-01, },
    {(float)+5.00579240e-01, (float)+9.97457828e-01, (float)+7.62856588e-01, (float)+7.49143075e-01, (float)+3.94754228e-01, },
    {(float)+4.99884152e-01, (float)+9.97481587e-01, (float)+7.62080342e-01, (float)+7.48762219e-01, (float)+3.93895349e-01, },
    {(float)+4.99189064e-01, (float)+9.97481587e-01, (float)+7.61304095e-01, (float)+7.48508315e-01, (float)+3.93036469e-01, },
    {(float)+4.98493976e-01, (float)+9.97481587e-01, (float)+7.60527848e-01, (float)+7.48254412e-01, (float)+3.92177590e-01, },
    {(float)+4.97798888e-01, (float)+9.97505346e-01, (float)+7.59557539e-01, (float)+7.47873556e-01, (float)+3.91318710e-01, },
    {(float)+4.97103800e-01, (float)+9.97505346e-01, (float)+7.58781292e-01, (float)+7.47619652e-01, (float)+3.90459831e-01, },
    {(float)+4.96408712e-01, (float)+9.97505346e-01, (float)+7.58005046e-01, (float)+7.47365748e-01, (float)+3.89584434e-01, },
    {(float)+4.95713624e-01, (float)+9.97529104e-01, (float)+7.57228799e-01, (float)+7.46984893e-01, (float)+3.88725555e-01, },
    {(float)+4.95018536e-01, (float)+9.97529104e-01, (float)+7.56452552e-01, (float)+7.46730989e-01, (float)+3.87866675e-01, },
    {(float)+4.94323448e-01, (float)+9.97529104e-01, (float)+7.55482243e-01, (float)+7.46350133e-01, (float)+3.87007796e-01, },
    {(float)+4.93628360e-01, (float)+9.97552863e-01, (float)+7.54705997e-01, (float)+7.46096230e-01, (float)+3.86148916e-01, },
    {(float)+4.92933272e-01, (float)+9.97552863e-01, (float)+7.53929750e-01, (float)+7.45842326e-01, (float)+3.85290037e-01, },
    {(float)+4.92238184e-01, (float)+9.97552863e-01, (float)+7.53153503e-01, (float)+7.45461470e-01, (float)+3.84431158e-01, },
    {(float)+4.91543095e-01, (float)+9.97576622e-01, (float)+7.52377256e-01, (float)+7.45207566e-01, (float)+3.83555761e-01, },
    {(float)+4.90848007e-01, (float)+9.97576622e-01, (float)+7.51406947e-01, (float)+7.44826711e-01, (float)+3.82696882e-01, },
    {(float)+4.90152919e-01, (float)+9.97576622e-01, (float)+7.50630701e-01, (float)+7.44572807e-01, (float)+3.81838002e-01, },
    {(float)+4.89457831e-01, (float)+9.97600380e-01, (float)+7.49854454e-01, (float)+7.44318903e-01, (float)+3.80979123e-01, },
    {(float)+4.88762743e-01, (float)+9.97600380e-01, (float)+7.49078207e-01, (float)+7.43938047e-01, (float)+3.80120243e-01, },
    {(float)+4.88067655e-01, (float)+9.97624139e-01, (float)+7.48301960e-01, (float)+7.43684144e-01, (float)+3.79261364e-01, },
    {(float)+4.87372567e-01, (float)+9.97624139e-01, (float)+7.47525713e-01, (float)+7.43430240e-01, (float)+3.78402484e-01, },
    {(float)+4.86677479e-01, (float)+9.97624139e-01, (float)+7.46555405e-01, (float)+7.43049384e-01, (float)+3.77543605e-01, },
    {(float)+4.85982391e-01, (float)+9.97647897e-01, (float)+7.45779158e-01, (float)+7.42795481e-01, (float)+3.76668208e-01, },
    {(float)+4.85287303e-01, (float)+9.97647897e-01, (float)+7.45002911e-01, (float)+7.42414625e-01, (float)+3.75809329e-01, },
    {(float)+4.84592215e-01, (float)+9.97647897e-01, (float)+7.44226664e-01, (float)+7.42160721e-01, (float)+3.74950449e-01, },
    {(float)+4.83897127e-01, (float)+9.97671656e-01, (float)+7.43450417e-01, (float)+7.41906817e-01, (float)+3.74091570e-01, },
    {(float)+4.83202039e-01, (float)+9.97671656e-01, (float)+7.42480109e-01, (float)+7.41525962e-01, (float)+3.73232690e-01, },
    {(float)+4.82506951e-01, (float)+9.97671656e-01, (float)+7.41703862e-01, (float)+7.41272058e-01, (float)+3.72373811e-01, },
    {(float)+4.81811863e-01, (float)+9.97695415e-01, (float)+7.40927615e-01, (float)+7.40891202e-01, (float)+3.71514931e-01, },
    {(float)+4.81116775e-01, (float)+9.97695415e-01, (float)+7.40151368e-01, (float)+7.40637298e-01, (float)+3.70639535e-01, },
    {(float)+4.80421687e-01, (float)+9.97695415e-01, (float)+7.39375121e-01, (float)+7.40383395e-01, (float)+3.69780655e-01, },
    {(float)+4.79726599e-01, (float)+9.97719173e-01, (float)+7.38404813e-01, (float)+7.40002539e-01, (float)+3.68921776e-01, },
    {(float)+4.79031511e-01, (float)+9.97719173e-01, (float)+7.37628566e-01, (float)+7.39748635e-01, (float)+3.68062896e-01, },
    {(float)+4.78336423e-01, (float)+9.97719173e-01, (float)+7.36852319e-01, (float)+7.39494731e-01, (float)+3.67204017e-01, },
    {(float)+4.77641335e-01, (float)+9.97742932e-01, (float)+7.36076072e-01, (float)+7.39113876e-01, (float)+3.66345137e-01, },
    {(float)+4.76946247e-01, (float)+9.97742932e-01, (float)+7.35299825e-01, (float)+7.38859972e-01, (float)+3.65486258e-01, },
    {(float)+4.76251158e-01, (float)+9.97742932e-01, (float)+7.34329517e-01, (float)+7.38479116e-01, (float)+3.64610862e-01, },
    {(float)+4.75556070e-01, (float)+9.97766690e-01, (float)+7.33553270e-01, (float)+7.38225213e-01, (float)+3.63751982e-01, },
    {(float)+4.74860982e-01, (float)+9.97766690e-01, (float)+7.32777023e-01, (float)+7.37971309e-01, (float)+3.62893103e-01, },
    {(float)+4.74165894e-01, (float)+9.97766690e-01, (float)+7.32000776e-01, (float)+7.37590453e-01, (float)+3.62034223e-01, },
    {(float)+4.73470806e-01, (float)+9.97790449e-01, (float)+7.31224529e-01, (float)+7.37336549e-01, (float)+3.61175344e-01, },
    {(float)+4.72775718e-01, (float)+9.97790449e-01, (float)+7.30254221e-01, (float)+7.37082646e-01, (float)+3.60316464e-01, },
    {(float)+4.72080630e-01, (float)+9.97790449e-01, (float)+7.29477974e-01, (float)+7.36701790e-01, (float)+3.59457585e-01, },
    {(float)+4.71385542e-01, (float)+9.97814208e-01, (float)+7.28701727e-01, (float)+7.36447886e-01, (float)+3.58582188e-01, },
    {(float)+4.70690454e-01, (float)+9.97814208e-01, (float)+7.27925480e-01, (float)+7.36067031e-01, (float)+3.57723309e-01, },
    {(float)+4.69995366e-01, (float)+9.97814208e-01, (float)+7.27149233e-01, (float)+7.35813127e-01, (float)+3.56864429e-01, },
    {(float)+4.69300278e-01, (float)+9.97837966e-01, (float)+7.26178925e-01, (float)+7.35559223e-01, (float)+3.56005550e-01, },
    {(float)+4.68605190e-01, (float)+9.97837966e-01, (float)+7.25402678e-01, (float)+7.35178367e-01, (float)+3.55146670e-01, },
    {(float)+4.67910102e-01, (float)+9.97837966e-01, (float)+7.24626431e-01, (float)+7.34924464e-01, (float)+3.54287791e-01, },
    {(float)+4.67215014e-01, (float)+9.97861725e-01, (float)+7.23850184e-01, (float)+7.34543608e-01, (float)+3.53428911e-01, },
    {(float)+4.66519926e-01, (float)+9.97861725e-01, (float)+7.23073938e-01, (float)+7.34289704e-01, (float)+3.52553515e-01, },
    {(float)+4.65824838e-01, (float)+9.97861725e-01, (float)+7.22103629e-01, (float)+7.34035800e-01, (float)+3.51694635e-01, },
    {(float)+4.65129750e-01, (float)+9.97885483e-01, (float)+7.21327382e-01, (float)+7.33654945e-01, (float)+3.50835756e-01, },
    {(float)+4.64434662e-01, (float)+9.97885483e-01, (float)+7.20551135e-01, (float)+7.33401041e-01, (float)+3.49976876e-01, },
    {(float)+4.63739574e-01, (float)+9.97885483e-01, (float)+7.19774888e-01, (float)+7.33147137e-01, (float)+3.49117997e-01, },
    {(float)+4.62928638e-01, (float)+9.97909242e-01, (float)+7.18998642e-01, (float)+7.32766282e-01, (float)+3.48259117e-01, },
    {(float)+4.62233550e-01, (float)+9.97909242e-01, (float)+7.18028333e-01, (float)+7.32512378e-01, (float)+3.47400238e-01, },
    {(float)+4.61538462e-01, (float)+9.97909242e-01, (float)+7.17252086e-01, (float)+7.32131522e-01, (float)+3.46524841e-01, },
    {(float)+4.60843373e-01, (float)+9.97933001e-01, (float)+7.16475839e-01, (float)+7.31877618e-01, (float)+3.45665962e-01, },
    {(float)+4.60148285e-01, (float)+9.97933001e-01, (float)+7.15699592e-01, (float)+7.31623715e-01, (float)+3.44807082e-01, },
    {(float)+4.59453197e-01, (float)+9.97933001e-01, (float)+7.14923346e-01, (float)+7.31242859e-01, (float)+3.43948203e-01, },
    {(float)+4.58758109e-01, (float)+9.97956759e-01, (float)+7.13953037e-01, (float)+7.30988955e-01, (float)+3.43089323e-01, },
    {(float)+4.58063021e-01, (float)+9.97956759e-01, (float)+7.13176790e-01, (float)+7.30608100e-01, (float)+3.42230444e-01, },
    {(float)+4.57367933e-01, (float)+9.97956759e-01, (float)+7.12400543e-01, (float)+7.30354196e-01, (float)+3.41371564e-01, },
    {(float)+4.56672845e-01, (float)+9.97980518e-01, (float)+7.11624297e-01, (float)+7.30100292e-01, (float)+3.40496168e-01, },
    {(float)+4.55977757e-01, (float)+9.97980518e-01, (float)+7.10848050e-01, (float)+7.29719436e-01, (float)+3.39637289e-01, },
    {(float)+4.55282669e-01, (float)+9.97980518e-01, (float)+7.09877741e-01, (float)+7.29465533e-01, (float)+3.38778409e-01, },
    {(float)+4.54587581e-01, (float)+9.98004277e-01, (float)+7.09101494e-01, (float)+7.29211629e-01, (float)+3.37919530e-01, },
    {(float)+4.53892493e-01, (float)+9.98004277e-01, (float)+7.08325247e-01, (float)+7.28830773e-01, (float)+3.37060650e-01, },
    {(float)+4.53197405e-01, (float)+9.98004277e-01, (float)+7.07549001e-01, (float)+7.28576869e-01, (float)+3.36201771e-01, },
    {(float)+4.52502317e-01, (float)+9.98028035e-01, (float)+7.06772754e-01, (float)+7.28196014e-01, (float)+3.35342891e-01, },
    {(float)+4.51807229e-01, (float)+9.98028035e-01, (float)+7.05802445e-01, (float)+7.27942110e-01, (float)+3.34467495e-01, },
    {(float)+4.51112141e-01, (float)+9.98028035e-01, (float)+7.05026198e-01, (float)+7.27688206e-01, (float)+3.33608615e-01, },
    {(float)+4.50417053e-01, (float)+9.98051794e-01, (float)+7.04249951e-01, (float)+7.27307351e-01, (float)+3.32749736e-01, },
    {(float)+4.49721965e-01, (float)+9.98051794e-01, (float)+7.03473705e-01, (float)+7.27053447e-01, (float)+3.31890856e-01, },
    {(float)+4.49026877e-01, (float)+9.98051794e-01, (float)+7.02697458e-01, (float)+7.26799543e-01, (float)+3.31031977e-01, },
    {(float)+4.48331789e-01, (float)+9.98075552e-01, (float)+7.01727149e-01, (float)+7.26418687e-01, (float)+3.30173097e-01, },
    {(float)+4.47636701e-01, (float)+9.98075552e-01, (float)+7.00950902e-01, (float)+7.26164784e-01, (float)+3.29314218e-01, },
    {(float)+4.46941613e-01, (float)+9.98075552e-01, (float)+7.00174656e-01, (float)+7.25783928e-01, (float)+3.28455338e-01, },
    {(float)+4.46246525e-01, (float)+9.98099311e-01, (float)+6.99398409e-01, (float)+7.25530024e-01, (float)+3.27579942e-01, },
    {(float)+4.45551437e-01, (float)+9.98099311e-01, (float)+6.98622162e-01, (float)+7.25276120e-01, (float)+3.26721062e-01, },
    {(float)+4.44856348e-01, (float)+9.98099311e-01, (float)+6.97651853e-01, (float)+7.24895265e-01, (float)+3.25862183e-01, },
    {(float)+4.44161260e-01, (float)+9.98123070e-01, (float)+6.96875606e-01, (float)+7.24641361e-01, (float)+3.25003303e-01, },
    {(float)+4.43466172e-01, (float)+9.98123070e-01, (float)+6.96099360e-01, (float)+7.24260505e-01, (float)+3.24144424e-01, },
    {(float)+4.42771084e-01, (float)+9.98123070e-01, (float)+6.95323113e-01, (float)+7.24006601e-01, (float)+3.23285544e-01, },
    {(float)+4.42075996e-01, (float)+9.98146828e-01, (float)+6.94546866e-01, (float)+7.23752698e-01, (float)+3.22426665e-01, },
    {(float)+4.41380908e-01, (float)+9.98146828e-01, (float)+6.93770619e-01, (float)+7.23371842e-01, (float)+3.21551268e-01, },
    {(float)+4.40685820e-01, (float)+9.98146828e-01, (float)+6.92800310e-01, (float)+7.23117938e-01, (float)+3.20692389e-01, },
    {(float)+4.39990732e-01, (float)+9.98170587e-01, (float)+6.92024064e-01, (float)+7.22864035e-01, (float)+3.19833510e-01, },
    {(float)+4.39295644e-01, (float)+9.98170587e-01, (float)+6.91247817e-01, (float)+7.22483179e-01, (float)+3.18974630e-01, },
    {(float)+4.38600556e-01, (float)+9.98170587e-01, (float)+6.90471570e-01, (float)+7.22229275e-01, (float)+3.18115751e-01, },
    {(float)+4.37905468e-01, (float)+9.98194345e-01, (float)+6.89695323e-01, (float)+7.21848419e-01, (float)+3.17256871e-01, },
    {(float)+4.37210380e-01, (float)+9.98194345e-01, (float)+6.88725015e-01, (float)+7.21594516e-01, (float)+3.16397992e-01, },
    {(float)+4.36515292e-01, (float)+9.98194345e-01, (float)+6.87948768e-01, (float)+7.21340612e-01, (float)+3.15522595e-01, },
    {(float)+4.35820204e-01, (float)+9.98218104e-01, (float)+6.87172521e-01, (float)+7.20959756e-01, (float)+3.14663716e-01, },
    {(float)+4.35125116e-01, (float)+9.98218104e-01, (float)+6.86396274e-01, (float)+7.20705852e-01, (float)+3.13804836e-01, },
    {(float)+4.34430028e-01, (float)+9.98218104e-01, (float)+6.85620027e-01, (float)+7.20324997e-01, (float)+3.12945957e-01, },
    {(float)+4.33734940e-01, (float)+9.98241863e-01, (float)+6.84649719e-01, (float)+7.20071093e-01, (float)+3.12087077e-01, },
    {(float)+4.33039852e-01, (float)+9.98241863e-01, (float)+6.83873472e-01, (float)+7.19817189e-01, (float)+3.11228198e-01, },
    {(float)+4.32344764e-01, (float)+9.98241863e-01, (float)+6.83097225e-01, (float)+7.19436334e-01, (float)+3.10369318e-01, },
    {(float)+4.31649676e-01, (float)+9.98265621e-01, (float)+6.82320978e-01, (float)+7.19182430e-01, (float)+3.09493922e-01, },
    {(float)+4.30954588e-01, (float)+9.98265621e-01, (float)+6.81544731e-01, (float)+7.18928526e-01, (float)+3.08635042e-01, },
    {(float)+4.30259500e-01, (float)+9.98265621e-01, (float)+6.80574423e-01, (float)+7.18547670e-01, (float)+3.07776163e-01, },
    {(float)+4.29564411e-01, (float)+9.98289380e-01, (float)+6.79798176e-01, (float)+7.18293767e-01, (float)+3.06917283e-01, },
    {(float)+4.28869323e-01, (float)+9.98289380e-01, (float)+6.79021929e-01, (float)+7.17912911e-01, (float)+3.06058404e-01, },
    {(float)+4.28174235e-01, (float)+9.98313139e-01, (float)+6.78245682e-01, (float)+7.17659007e-01, (float)+3.05199524e-01, },
    {(float)+4.27479147e-01, (float)+9.98313139e-01, (float)+6.77469435e-01, (float)+7.17405103e-01, (float)+3.04340645e-01, },
    {(float)+4.26784059e-01, (float)+9.98313139e-01, (float)+6.76499127e-01, (float)+7.17024248e-01, (float)+3.03465248e-01, },
    {(float)+4.26088971e-01, (float)+9.98336897e-01, (float)+6.75722880e-01, (float)+7.16770344e-01, (float)+3.02606369e-01, },
    {(float)+4.25393883e-01, (float)+9.98336897e-01, (float)+6.74946633e-01, (float)+7.16516440e-01, (float)+3.01747489e-01, },
    {(float)+4.24698795e-01, (float)+9.98336897e-01, (float)+6.74170386e-01, (float)+7.16135585e-01, (float)+3.00888610e-01, },
    {(float)+4.24003707e-01, (float)+9.98360656e-01, (float)+6.73394139e-01, (float)+7.15881681e-01, (float)+3.00029730e-01, },
    {(float)+4.23308619e-01, (float)+9.98360656e-01, (float)+6.72423831e-01, (float)+7.15500825e-01, (float)+2.99170851e-01, },
    {(float)+4.22613531e-01, (float)+9.98360656e-01, (float)+6.71647584e-01, (float)+7.15246921e-01, (float)+2.98311971e-01, },
    {(float)+4.21918443e-01, (float)+9.98384414e-01, (float)+6.70871337e-01, (float)+7.14993018e-01, (float)+2.97436575e-01, },
    {(float)+4.21223355e-01, (float)+9.98384414e-01, (float)+6.70095090e-01, (float)+7.14612162e-01, (float)+2.96577696e-01, },
    {(float)+4.20528267e-01, (float)+9.98384414e-01, (float)+6.69318843e-01, (float)+7.14358258e-01, (float)+2.95718816e-01, },
    {(float)+4.19833179e-01, (float)+9.98408173e-01, (float)+6.68348535e-01, (float)+7.13977403e-01, (float)+2.94859937e-01, },
    {(float)+4.19138091e-01, (float)+9.98408173e-01, (float)+6.67572288e-01, (float)+7.13723499e-01, (float)+2.94001057e-01, },
    {(float)+4.18443003e-01, (float)+9.98408173e-01, (float)+6.66796041e-01, (float)+7.13469595e-01, (float)+2.93142178e-01, },
    {(float)+4.17747915e-01, (float)+9.98431932e-01, (float)+6.66019794e-01, (float)+7.13088739e-01, (float)+2.92283298e-01, },
    {(float)+4.17052827e-01, (float)+9.98431932e-01, (float)+6.65243547e-01, (float)+7.12834836e-01, (float)+2.91407902e-01, },
    {(float)+4.16357739e-01, (float)+9.98431932e-01, (float)+6.64273239e-01, (float)+7.12580932e-01, (float)+2.90549022e-01, },
    {(float)+4.15662651e-01, (float)+9.98455690e-01, (float)+6.63496992e-01, (float)+7.12200076e-01, (float)+2.89690143e-01, },
    {(float)+4.14967563e-01, (float)+9.98455690e-01, (float)+6.62720745e-01, (float)+7.11946172e-01, (float)+2.88831263e-01, },
    {(float)+4.14272475e-01, (float)+9.98455690e-01, (float)+6.61944498e-01, (float)+7.11565317e-01, (float)+2.87972384e-01, },
    {(float)+4.13577386e-01, (float)+9.98479449e-01, (float)+6.61168252e-01, (float)+7.11311413e-01, (float)+2.87113504e-01, },
    {(float)+4.12882298e-01, (float)+9.98479449e-01, (float)+6.60197943e-01, (float)+7.11057509e-01, (float)+2.86254625e-01, },
    {(float)+4.12187210e-01, (float)+9.98479449e-01, (float)+6.59421696e-01, (float)+7.10676654e-01, (float)+2.85395745e-01, },
    {(float)+4.11492122e-01, (float)+9.98503207e-01, (float)+6.58645449e-01, (float)+7.10422750e-01, (float)+2.84520349e-01, },
    {(float)+4.10681186e-01, (float)+9.98503207e-01, (float)+6.57869202e-01, (float)+7.10041894e-01, (float)+2.83661469e-01, },
    {(float)+4.09986098e-01, (float)+9.98503207e-01, (float)+6.57092956e-01, (float)+7.09787990e-01, (float)+2.82802590e-01, },
    {(float)+4.09291010e-01, (float)+9.98526966e-01, (float)+6.56122647e-01, (float)+7.09534087e-01, (float)+2.81943710e-01, },
    {(float)+4.08595922e-01, (float)+9.98526966e-01, (float)+6.55346400e-01, (float)+7.09153231e-01, (float)+2.81084831e-01, },
    {(float)+4.07900834e-01, (float)+9.98526966e-01, (float)+6.54570153e-01, (float)+7.08899327e-01, (float)+2.80225951e-01, },
    {(float)+4.07205746e-01, (float)+9.98550725e-01, (float)+6.53793906e-01, (float)+7.08645423e-01, (float)+2.79367072e-01, },
    {(float)+4.06510658e-01, (float)+9.98550725e-01, (float)+6.53017660e-01, (float)+7.08264568e-01, (float)+2.78491675e-01, },
    {(float)+4.05815570e-01, (float)+9.98550725e-01, (float)+6.52047351e-01, (float)+7.08010664e-01, (float)+2.77632796e-01, },
    {(float)+4.05120482e-01, (float)+9.98574483e-01, (float)+6.51271104e-01, (float)+7.07629808e-01, (float)+2.76773916e-01, },
    {(float)+4.04425394e-01, (float)+9.98574483e-01, (float)+6.50494857e-01, (float)+7.07375905e-01, (float)+2.75915037e-01, },
    {(float)+4.03730306e-01, (float)+9.98574483e-01, (float)+6.49718611e-01, (float)+7.07122001e-01, (float)+2.75056158e-01, },
    {(float)+4.03035218e-01, (float)+9.98598242e-01, (float)+6.48942364e-01, (float)+7.06741145e-01, (float)+2.74197278e-01, },
    {(float)+4.02340130e-01, (float)+9.98598242e-01, (float)+6.47972055e-01, (float)+7.06487241e-01, (float)+2.73338399e-01, },
    {(float)+4.01645042e-01, (float)+9.98598242e-01, (float)+6.47195808e-01, (float)+7.06233338e-01, (float)+2.72463002e-01, },
    {(float)+4.00949954e-01, (float)+9.98622000e-01, (float)+6.46419561e-01, (float)+7.05852482e-01, (float)+2.71604123e-01, },
    {(float)+4.00254866e-01, (float)+9.98622000e-01, (float)+6.45643315e-01, (float)+7.05598578e-01, (float)+2.70745243e-01, },
    {(float)+3.99559778e-01, (float)+9.98622000e-01, (float)+6.44867068e-01, (float)+7.05217722e-01, (float)+2.69886364e-01, },
    {(float)+3.98864690e-01, (float)+9.98645759e-01, (float)+6.43896759e-01, (float)+7.04963819e-01, (float)+2.69027484e-01, },
    {(float)+3.98169601e-01, (float)+9.98645759e-01, (float)+6.43120512e-01, (float)+7.04709915e-01, (float)+2.68168605e-01, },
    {(float)+3.97474513e-01, (float)+9.98645759e-01, (float)+6.42344265e-01, (float)+7.04329059e-01, (float)+2.67309725e-01, },
    {(float)+3.96779425e-01, (float)+9.98669518e-01, (float)+6.41568019e-01, (float)+7.04075156e-01, (float)+2.66434329e-01, },
    {(float)+3.96084337e-01, (float)+9.98669518e-01, (float)+6.40791772e-01, (float)+7.03694300e-01, (float)+2.65575449e-01, },
    {(float)+3.95389249e-01, (float)+9.98669518e-01, (float)+6.40015525e-01, (float)+7.03440396e-01, (float)+2.64716570e-01, },
    {(float)+3.94694161e-01, (float)+9.98693276e-01, (float)+6.39045216e-01, (float)+7.03186492e-01, (float)+2.63857690e-01, },
    {(float)+3.93999073e-01, (float)+9.98693276e-01, (float)+6.38268970e-01, (float)+7.02805637e-01, (float)+2.62998811e-01, },
    {(float)+3.93303985e-01, (float)+9.98693276e-01, (float)+6.37492723e-01, (float)+7.02551733e-01, (float)+2.62139931e-01, },
    {(float)+3.92608897e-01, (float)+9.98717035e-01, (float)+6.36716476e-01, (float)+7.02297829e-01, (float)+2.61281052e-01, },
    {(float)+3.91913809e-01, (float)+9.98717035e-01, (float)+6.35940229e-01, (float)+7.01916973e-01, (float)+2.60405655e-01, },
    {(float)+3.91218721e-01, (float)+9.98717035e-01, (float)+6.34969920e-01, (float)+7.01663070e-01, (float)+2.59546776e-01, },
    {(float)+3.90523633e-01, (float)+9.98740794e-01, (float)+6.34193674e-01, (float)+7.01282214e-01, (float)+2.58687896e-01, },
    {(float)+3.89828545e-01, (float)+9.98740794e-01, (float)+6.33417427e-01, (float)+7.01028310e-01, (float)+2.57829017e-01, },
    {(float)+3.89133457e-01, (float)+9.98740794e-01, (float)+6.32641180e-01, (float)+7.00774406e-01, (float)+2.56970137e-01, },
    {(float)+3.88438369e-01, (float)+9.98764552e-01, (float)+6.31864933e-01, (float)+7.00393551e-01, (float)+2.56111258e-01, },
    {(float)+3.87743281e-01, (float)+9.98764552e-01, (float)+6.30894624e-01, (float)+7.00139647e-01, (float)+2.55252378e-01, },
    {(float)+3.87048193e-01, (float)+9.98764552e-01, (float)+6.30118378e-01, (float)+6.99758791e-01, (float)+2.54376982e-01, },
    {(float)+3.86353105e-01, (float)+9.98788311e-01, (float)+6.29342131e-01, (float)+6.99504888e-01, (float)+2.53518103e-01, },
    {(float)+3.85658017e-01, (float)+9.98788311e-01, (float)+6.28565884e-01, (float)+6.99250984e-01, (float)+2.52659223e-01, },
    {(float)+3.84962929e-01, (float)+9.98788311e-01, (float)+6.27789637e-01, (float)+6.98870128e-01, (float)+2.51800344e-01, },
    {(float)+3.84267841e-01, (float)+9.98812069e-01, (float)+6.26819329e-01, (float)+6.98616224e-01, (float)+2.50941464e-01, },
    {(float)+3.83572753e-01, (float)+9.98812069e-01, (float)+6.26043082e-01, (float)+6.98362321e-01, (float)+2.50082585e-01, },
    {(float)+3.82877665e-01, (float)+9.98812069e-01, (float)+6.25266835e-01, (float)+6.97981465e-01, (float)+2.49223705e-01, },
    {(float)+3.82182576e-01, (float)+9.98835828e-01, (float)+6.24490588e-01, (float)+6.97727561e-01, (float)+2.48348309e-01, },
    {(float)+3.81487488e-01, (float)+9.98835828e-01, (float)+6.23714341e-01, (float)+6.97346706e-01, (float)+2.47489429e-01, },
    {(float)+3.80792400e-01, (float)+9.98835828e-01, (float)+6.22744033e-01, (float)+6.97092802e-01, (float)+2.46630550e-01, },
    {(float)+3.80097312e-01, (float)+9.98859587e-01, (float)+6.21967786e-01, (float)+6.96838898e-01, (float)+2.45771670e-01, },
    {(float)+3.79402224e-01, (float)+9.98859587e-01, (float)+6.21191539e-01, (float)+6.96458042e-01, (float)+2.44912791e-01, },
    {(float)+3.78707136e-01, (float)+9.98859587e-01, (float)+6.20415292e-01, (float)+6.96204139e-01, (float)+2.44053911e-01, },
    {(float)+3.78012048e-01, (float)+9.98883345e-01, (float)+6.19639045e-01, (float)+6.95950235e-01, (float)+2.43195032e-01, },
    {(float)+3.77316960e-01, (float)+9.98883345e-01, (float)+6.18668737e-01, (float)+6.95569379e-01, (float)+2.42319635e-01, },
    {(float)+3.76621872e-01, (float)+9.98883345e-01, (float)+6.17892490e-01, (float)+6.95315475e-01, (float)+2.41460756e-01, },
    {(float)+3.75926784e-01, (float)+9.98907104e-01, (float)+6.17116243e-01, (float)+6.94934620e-01, (float)+2.40601876e-01, },
    {(float)+3.75231696e-01, (float)+9.98907104e-01, (float)+6.16339996e-01, (float)+6.94680716e-01, (float)+2.39742997e-01, },
    {(float)+3.74536608e-01, (float)+9.98907104e-01, (float)+6.15563749e-01, (float)+6.94426812e-01, (float)+2.38884117e-01, },
    {(float)+3.73841520e-01, (float)+9.98930862e-01, (float)+6.14593441e-01, (float)+6.94045957e-01, (float)+2.38025238e-01, },
    {(float)+3.73146432e-01, (float)+9.98930862e-01, (float)+6.13817194e-01, (float)+6.93792053e-01, (float)+2.37166358e-01, },
    {(float)+3.72451344e-01, (float)+9.98930862e-01, (float)+6.13040947e-01, (float)+6.93411197e-01, (float)+2.36307479e-01, },
    {(float)+3.71756256e-01, (float)+9.98954621e-01, (float)+6.12264700e-01, (float)+6.93157293e-01, (float)+2.35432082e-01, },
    {(float)+3.71061168e-01, (float)+9.98954621e-01, (float)+6.11488453e-01, (float)+6.92903390e-01, (float)+2.34573203e-01, },
    {(float)+3.70366080e-01, (float)+9.98954621e-01, (float)+6.10518145e-01, (float)+6.92522534e-01, (float)+2.33714323e-01, },
    {(float)+3.69670992e-01, (float)+9.98978380e-01, (float)+6.09741898e-01, (float)+6.92268630e-01, (float)+2.32855444e-01, },
    {(float)+3.68975904e-01, (float)+9.98978380e-01, (float)+6.08965651e-01, (float)+6.92014726e-01, (float)+2.31996564e-01, },
    {(float)+3.68280816e-01, (float)+9.99002138e-01, (float)+6.08189404e-01, (float)+6.91633871e-01, (float)+2.31137685e-01, },
    {(float)+3.67585728e-01, (float)+9.99002138e-01, (float)+6.07413157e-01, (float)+6.91379967e-01, (float)+2.30278805e-01, },
    {(float)+3.66890639e-01, (float)+9.99002138e-01, (float)+6.06442849e-01, (float)+6.90999111e-01, (float)+2.29403409e-01, },
    {(float)+3.66195551e-01, (float)+9.99025897e-01, (float)+6.05666602e-01, (float)+6.90745208e-01, (float)+2.28544530e-01, },
    {(float)+3.65500463e-01, (float)+9.99025897e-01, (float)+6.04890355e-01, (float)+6.90491304e-01, (float)+2.27685650e-01, },
    {(float)+3.64805375e-01, (float)+9.99025897e-01, (float)+6.04114108e-01, (float)+6.90110448e-01, (float)+2.26826771e-01, },
    {(float)+3.64110287e-01, (float)+9.99049656e-01, (float)+6.03337861e-01, (float)+6.89856544e-01, (float)+2.25967891e-01, },
    {(float)+3.63415199e-01, (float)+9.99049656e-01, (float)+6.02367553e-01, (float)+6.89475689e-01, (float)+2.25109012e-01, },
    {(float)+3.62720111e-01, (float)+9.99049656e-01, (float)+6.01591306e-01, (float)+6.89221785e-01, (float)+2.24250132e-01, },
    {(float)+3.62025023e-01, (float)+9.99073414e-01, (float)+6.00815059e-01, (float)+6.88967881e-01, (float)+2.23374736e-01, },
    {(float)+3.61329935e-01, (float)+9.99073414e-01, (float)+6.00038812e-01, (float)+6.88587026e-01, (float)+2.22515856e-01, },
    {(float)+3.60634847e-01, (float)+9.99073414e-01, (float)+5.99262565e-01, (float)+6.88333122e-01, (float)+2.21656977e-01, },
    {(float)+3.59939759e-01, (float)+9.99097173e-01, (float)+5.98292257e-01, (float)+6.88079218e-01, (float)+2.20798097e-01, },
    {(float)+3.59244671e-01, (float)+9.99097173e-01, (float)+5.97516010e-01, (float)+6.87698362e-01, (float)+2.19939218e-01, },
    {(float)+3.58433735e-01, (float)+9.99097173e-01, (float)+5.96739763e-01, (float)+6.87444459e-01, (float)+2.19080338e-01, },
    {(float)+3.57738647e-01, (float)+9.99120931e-01, (float)+5.95963516e-01, (float)+6.87063603e-01, (float)+2.18221459e-01, },
    {(float)+3.57043559e-01, (float)+9.99120931e-01, (float)+5.95187270e-01, (float)+6.86809699e-01, (float)+2.17346062e-01, },
    {(float)+3.56348471e-01, (float)+9.99120931e-01, (float)+5.94216961e-01, (float)+6.86555795e-01, (float)+2.16487183e-01, },
    {(float)+3.55653383e-01, (float)+9.99144690e-01, (float)+5.93440714e-01, (float)+6.86174940e-01, (float)+2.15628303e-01, },
    {(float)+3.54958295e-01, (float)+9.99144690e-01, (float)+5.92664467e-01, (float)+6.85921036e-01, (float)+2.14769424e-01, },
    {(float)+3.54263207e-01, (float)+9.99144690e-01, (float)+5.91888220e-01, (float)+6.85667132e-01, (float)+2.13910544e-01, },
    {(float)+3.53568119e-01, (float)+9.99168449e-01, (float)+5.91111974e-01, (float)+6.85286277e-01, (float)+2.13051665e-01, },
    {(float)+3.52873031e-01, (float)+9.99168449e-01, (float)+5.90141665e-01, (float)+6.85032373e-01, (float)+2.12192785e-01, },
    {(float)+3.52177943e-01, (float)+9.99168449e-01, (float)+5.89365418e-01, (float)+6.84651517e-01, (float)+2.11317389e-01, },
    {(float)+3.51482854e-01, (float)+9.99192207e-01, (float)+5.88589171e-01, (float)+6.84397613e-01, (float)+2.10458510e-01, },
    {(float)+3.50787766e-01, (float)+9.99192207e-01, (float)+5.87812925e-01, (float)+6.84143710e-01, (float)+2.09599630e-01, },
    {(float)+3.50092678e-01, (float)+9.99192207e-01, (float)+5.87036678e-01, (float)+6.83762854e-01, (float)+2.08740751e-01, },
    {(float)+3.49397590e-01, (float)+9.99215966e-01, (float)+5.86260431e-01, (float)+6.83508950e-01, (float)+2.07881871e-01, },
    {(float)+3.48702502e-01, (float)+9.99215966e-01, (float)+5.85290122e-01, (float)+6.83128094e-01, (float)+2.07022992e-01, },
    {(float)+3.48007414e-01, (float)+9.99215966e-01, (float)+5.84513875e-01, (float)+6.82874191e-01, (float)+2.06164112e-01, },
    {(float)+3.47312326e-01, (float)+9.99239724e-01, (float)+5.83737629e-01, (float)+6.82620287e-01, (float)+2.05288716e-01, },
    {(float)+3.46617238e-01, (float)+9.99239724e-01, (float)+5.82961382e-01, (float)+6.82239431e-01, (float)+2.04429836e-01, },
    {(float)+3.45922150e-01, (float)+9.99239724e-01, (float)+5.82185135e-01, (float)+6.81985527e-01, (float)+2.03570957e-01, },
    {(float)+3.45227062e-01, (float)+9.99263483e-01, (float)+5.81214826e-01, (float)+6.81731624e-01, (float)+2.02712077e-01, },
    {(float)+3.44531974e-01, (float)+9.99263483e-01, (float)+5.80438579e-01, (float)+6.81350768e-01, (float)+2.01853198e-01, },
    };
    vector<vector<float>> transpose_result = transposeMatrix(initial_data);
    // cout << "转置结果：" << transpose_result.size() << " " << transpose_result[0].size() << endl;
    vector<vector<float>> input_data = transpose_result;

    if (input_data.size() != transpose_result.size())return -1;

    vector<vector<float>> target = transpose_result;

    int function_type =1;//Adam算法不能用xigmoid,要用tanh，sigmoid
    int node = 25;//第一层节点20,25和10组合
    int laye2_neural_nodeCount = 10;//5
    int epochs = 5;//训练轮次

    int target_mse = 0.000001;
    float lr = 0.0001;//用tanh展平函数学习率小一点好3配合0.001

    // 定义第一层
    auto result_tuple_layer1 = initial_neurallayer_output(input_data, node, function_type);
    // 从 tuple 中获取返回的值
    vector<vector<float>> layer1_output = get<0>(result_tuple_layer1);
    vector<float> layer1_biasmatrix = get<1>(result_tuple_layer1);
    vector<vector<float>> layel1_weightmatrix = get<2>(result_tuple_layer1);
    cout << "row" << layel1_weightmatrix.size() << endl <<"culs" << layel1_weightmatrix[0].size() << endl;
    //Print printer;
    //printer.print(layel1_weightmatrix);
    vector<vector<float>> layer1_output_ini(layer1_output.size(), vector<float>(layer1_output[0].size(), 0));
    vector<vector<float>> m_1;
    vector<vector<float>> v_1;
    initialize_m_v(m_1, v_1, layel1_weightmatrix.size(), layel1_weightmatrix[0].size());
    
    // 定义第二层
    auto result_tuple_layer2 = initial_neurallayer_output(layer1_output, laye2_neural_nodeCount, function_type);
    vector<vector<float>> layer2_output = get<0>(result_tuple_layer2);
    vector<float> layer2_biasmatrix = get<1>(result_tuple_layer2);
    vector<vector<float>> layel2_weightmatrix = get<2>(result_tuple_layer2);

    vector<vector<float>> layer2_output_ini(layer2_output.size(), vector<float>(layer2_output[0].size(), 0));
    vector<vector<float>> m_2;
    vector<vector<float>> v_2;
    initialize_m_v(m_2, v_2, layel2_weightmatrix.size(), layel2_weightmatrix[0].size());


    // 输出层
    auto result_tuple_op = initial_neurallayer_output(layer2_output, target.size(), function_type);
    vector<vector<float>> output = get<0>(result_tuple_op);
    vector<float> op_biasmatrix = get<1>(result_tuple_op);
    vector<vector<float>> op_weightmatrix = get<2>(result_tuple_op);
    vector<vector<float>> error = calculateerror(output, target);
    vector<vector<float>> m_o;
    vector<vector<float>> v_o;
    initialize_m_v(m_o, v_o, op_weightmatrix.size(), op_weightmatrix[0].size());



    float mse = calculateMSE(error);
    vector<vector<float>> layer2_error = calculateerror(layer2_output, layer2_output_ini);
    vector<vector<float>> layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出
    //print2DArray(op_weightmatrix);
    //神经网络训练主循环
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        layer1_output = training_forward_neurallayer_output(input_data, layel1_weightmatrix, layer1_biasmatrix, node, function_type);

        // 第二层
        layer2_output = training_forward_neurallayer_output(layer1_output, layel2_weightmatrix, layer2_biasmatrix, laye2_neural_nodeCount, function_type);

        // 输出层
        output = training_forward_neurallayer_output(layer2_output, op_weightmatrix, op_biasmatrix, target.size(), function_type);

        error = calculateerror(output, target);
        mse = calculateMSE(error);
        layer2_error = calculateerror(layer2_output, layer2_output_ini);
        layer1_error = calculateerror(layer1_output, layer1_output_ini);//要将上一步输出赋予初始输出
        //update_weights_bias(op_weightmatrix, op_biasmatrix, output, error, layer2_output, lr);//参数顺序：该层权重，该层输出，该层误差，该层输入，学习率
        //update_weights_bias(layel2_weightmatrix, layer2_biasmatrix, layer2_output, layer2_error, layer1_output, lr);
       //update_weights_bias(layel1_weightmatrix, layer1_biasmatrix, layer1_output, layer1_error, input_data, lr);
        //adam_optimizer(op_weightmatrix, output, error, layer2_output,m_o,v_o,epoch,lr);
        //adam_optimizer(layel2_weightmatrix, layer2_output, layer2_error, layer1_output, m_2, v_2, epoch, lr);
        //adam_optimizer(layel1_weightmatrix, layer1_output, layer1_error, input_data, m_1, v_1, epoch, lr);
        adam_optimizer_bias(op_weightmatrix, op_biasmatrix,output, error, layer2_output, m_o, v_o, epoch ,lr);
        adam_optimizer_bias(layel2_weightmatrix, layer2_biasmatrix,layer2_output, layer2_error, layer1_output, m_2, v_2, epoch ,lr);
        adam_optimizer_bias(layel1_weightmatrix, layer1_biasmatrix,layer1_output, layer1_error, input_data, m_1, v_1, epoch ,lr);
        //print2DArray(m_o);
        //print2DArray(v_o);
        //print1DArray(op_biasmatrix);
        cout << "训练步数：" << epoch << "  " << "MSE:" << mse << endl;
        //lr = lr * exp(-0.1*epoch);
        //print2DArray(op_weightmatrix);

        if (mse < target_mse) break;

    }
    
    //cout << "最后一次迭代输出结果：" << endl;
    //print2DArray(output);

    



    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
