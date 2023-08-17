#include<math.h>
#include<stdio.h>
#include<omp.h>

double devided, gauss_coeff, cs_coeff;  // , gamma;
long whole_num_point, state_num;
double *E_list, *f_list;

double gauss(double E, double dE_0n){
    // devided = -delta * delta / 2;
    double diff = E - dE_0n;
    double to_be_exp = diff * diff / devided;
    // return gauss_coeff * exp(to_be_exp);
    double tmp = gauss_coeff * exp(to_be_exp);
    // printf("tmp: %f\n", tmp);
    return tmp;
}

double getValue(double *arr, long iRow, long jCol){
    return *(arr + iRow*whole_num_point + jCol);
}

void calc_devid_para(double delta){
    // printf("delta value: %f\n", delta);
    devided = -delta * delta / 2;
}

double calc_at_de(double de){
    double spect = 0.0;
    double E;
    // printf("state_num: %d\t whole_num_point: %d\n", state_num, whole_num_point);
    for (long state=0; state < state_num; state++){
        for (long idx=0; idx < whole_num_point; idx++){
            //printf("before getValue\n");
            E = getValue(E_list, state, idx);
            // printf("after getValue\n");
            // printf("state: %d \t E value: %f\n", state, E);
            // E = E_list[state][idx];
            // E = *(E_list * state + idx)
            // spect += E * f_list[state][idx] * gauss(de, E);
            spect += (E * getValue(f_list, state, idx) * gauss(de, E));
        }
    }
    double cs = cs_coeff * spect / (whole_num_point * de);
    return cs;
    // return coeff * spect / (whole_num_point * de);
}

extern "C" void cs_calc(double *py_E_list, double *py_f_list, 
            long py_state_num, long py_whole_num_point, 
            double py_delta, double py_gauss_coeff, long de_num, double py_cs_coeff, // double py_gamma, 
            double cross_section[], double de_list[]){
    double de;
    // long length = sizeof(de_list)/sizeof(de_list[0]);
    long length = de_num;
    // printf("length: %d", length);
    E_list = py_E_list;
    f_list = py_f_list;
    state_num = py_state_num;
    whole_num_point = py_whole_num_point;
    gauss_coeff = py_gauss_coeff;
    cs_coeff = py_cs_coeff;
    // gamma = py_gamma;
    calc_devid_para(py_delta);
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i < length; i++){
        de = de_list[i];
        cross_section[i] = calc_at_de(de);
        // printf("%f\t%f\n", de, cross_section[i]);
    }
}