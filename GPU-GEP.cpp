/*
2018-4-21
基于上一版本删除了cde，剩下了代理模型和带强化学习性质的知识迁移
新增了移动方案

该版本的移动方案是移动距离为两者中心点差距，感觉可能会有点不靠谱的样子

蚁群策略中的dis参数还是得考虑——使用了softmax对距离进行了对一化，可能效果会好一点

2018-5-17 修正了一些错误

2018-7-25 这才是最核心的版本

2018-7-27 基于上一个实验版本，本版本主要就是为这些函数加上旋转矩阵

2018-8-5 基于上一个版本，减少了信息素浓度的下限，使得可以减少与无用的函数进行知识迁移（FROM 0.1 TO 0.01）
		 减少搜索次数或者高斯过程参数搜索的过程

*/

#include<iostream>
#include<algorithm>
#include <vector>
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include<queue>
#include "string.h"
using namespace std;
const double PI = acos(-1);
const int MAX_FUNC_NUM = 6;
int func_num = 1;//当前问题数量
const int dim = 30;//问题维度
const int MAX_FIT_TIME = 5; //-------------------------xiugai--------------------------
const int maxn = 200;//单次训练集的最大点数
const int maxm = 60000;//总集最大点数
const double E = exp(1);

const double esp1 = 1e-6;//答案精度误差，小于视为收敛至最优值
const double inf = 1e20;//无穷大
const bool show_message = 1;//控制展示信息与否

double solution_space[MAX_FUNC_NUM][dim][2];//搜索空间

const double global_lbound = -1, global_rbound = 1;

bool end_f[MAX_FUNC_NUM];//问题收敛与否

int cnt_id;//当前正在训练的任务id



//============公用函数===========
struct NODE{
	double x[dim];
	double f;
	NODE(){}
	NODE(double *xx, double ff = 0){
		for (int i = 0; i < dim; i++)x[i] = xx[i];
		f = ff;
	}

	void init(){
		f = inf;
	}

	bool operator==(NODE b)const{
		for (int i = 0; i < dim; i++)if (this->x[i] != b.x[i])return 0;
		return 1;
	}

	friend double o_dis(NODE a, NODE b){
		double ans = 0;
		for (int i = 0; i < dim; i++){
			ans += a.x[i] - b.x[i];
		}
		return ans;
	}

	friend double dis(NODE a, NODE b){//计算距离-
		double ans = 0;
		for (int i = 0; i < dim; i++){
			ans += (a.x[i] - b.x[i])*(a.x[i] - b.x[i]);
		}
		return ans;
	}
};

NODE real_best_p[MAX_FUNC_NUM];
NODE cnt_round_p[MAX_FUNC_NUM];

static int phase = 0;
double gaussian()
{
	static double V1, V2, S;
	double X;
	if (phase == 0) {
		do {
			double U1 = (double)rand() / (double)RAND_MAX;
			double U2 = (double)rand() / (double)RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);
	return X;

	phase = 1 - phase;
}

double gauss(double a, double b)//随机从正态分布上抽样
{
	return a + gaussian() * b;
}

double cauchy(double location, double t)//从柯西分布上抽样
{
	double v1 = gauss(0, 1);
	double v2 = gauss(0, 1);
	if (v2 != 0)
	{
		return t * v1 / v2 + location;
	}
	return location;
}

double randval(double a, double b)//在[a,b]内产生随机数
{
	return a + (b - a) * rand() / (double)RAND_MAX;
}

double sqr(double x){
	return x*x;
}

int fit_time[MAX_FUNC_NUM];//真实评估次数

double map_parameter[MAX_FUNC_NUM] = { 50, 50, 50, 50, 50, 50};

NODE mapping(NODE &cnt, int id = cnt_id){//函数从[-1,1]映射回原空间
	NODE ret;
	for (int i = 0; i < dim; i++){
		ret.x[i] = cnt.x[i] * map_parameter[id];
	}
	return ret;
}


double power(double y, int k){
	double ans = 1;
	for (int i = 1; i <= k; i++)ans *= y;
	return ans;
}


double test_function(double *x, int id){
	double ans = 0;
	
	return ans;
}


double fitness_function(double *x, int cntid = cnt_id){//the real fitness
	double ans = 0;
	
	return ans;
}


const int MODEL_GENE_NVARS = 5;

const int Nc = 60;
const int Nr = 60;
int training_data_num[MAX_FUNC_NUM];
int tot_data_num[MAX_FUNC_NUM], rencent_point_num[MAX_FUNC_NUM];
NODE tot_data[MAX_FUNC_NUM][maxm];
NODE recent_data[MAX_FUNC_NUM][maxm];
NODE train_data[MAX_FUNC_NUM][maxn];

double sigma_f, l, sigma_n, f1, v;

struct gene_model{
	double x[MODEL_GENE_NVARS];
	double f;
	void init(){
		f = 1e50;
	}
}best_model[MAX_FUNC_NUM];

double K[maxn][maxn];//covariance matrix
double K_s[maxn][maxn];
double K_ss[maxn][maxn];

double inv_K[maxn][maxn];
double L[maxn][maxn];//下三角矩阵
double L_T[maxn][maxn];//上三角矩阵
double L_1[maxn][maxn];//下三角矩阵的逆
double L_T_1[maxn][maxn];//上三角矩阵的逆


namespace training {

	const int MAXEVALS = 1000;
	const int POPSIZE = 20;
	double  cp = 0.1;
	double  cC = 0.1;
	int evals;
	double	LBOUND, UBOUND;//超参数搜索空间

	gene_model population[POPSIZE], newpopulation[POPSIZE], u_population[POPSIZE];
	int archive_size;
	gene_model archives[POPSIZE];

	inline double kernel_function(NODE& x, NODE&x2){
		double cnt;
		cnt = sigma_f * sigma_f * exp(dis(x, x2) / (-2 * l * l)) + f1*f1* exp(-2 * sin(v * PI * o_dis(x, x2)) * sin(v * PI *o_dis(x, x2)));
		//cnt = sigma_f * sigma_f * exp(olf_dis(x, x2) / (-2 * l*l));                   //
		//cnt = cnt = sigma_f * sigma_f * pow(1 + o_dis(x, x2) * o_dis(x, x2) / (2 * l * v * v), -l);
		return cnt;
	}

	double error_function(NODE &x, NODE &x2){
		if (x == x2)
			return sigma_n*sigma_n;
		return 0;
	}

	double ker(NODE &x, NODE &x2){
		return kernel_function(x, x2) + error_function(x, x2);
	}

	void generate_K(){
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			//printf("training_data_num[cnt_id]=%d\n", training_data_num[cnt_id]);
			for (int j = 0; j < training_data_num[cnt_id]; j++){
				K[i][j] = ker(train_data[cnt_id][i], train_data[cnt_id][j]);
			}
		}
	}

	int chol(double a[][maxn], int n, double *det)//chol矩阵分解
	{
		int i, j, k, u, v, l;
		double d;
		if ((a[0][0] + 1.0 == 1.0) || (a[0][0] < 0.0))
		{
			//printf("fail\n"); 
			return 0;
		}
		a[0][0] = sqrt(a[0][0]);
		d = a[0][0];
		for (i = 1; i <= n - 1; i++)
		{
			u = i*n; a[i][0] = a[i][0] / a[0][0];
		}
		for (j = 1; j <= n - 1; j++)
		{
			l = j*n + j;
			for (k = 0; k <= j - 1; k++)
			{
				u = j*n + k; a[j][j] = a[j][j] - a[j][k] * a[j][k];
			}
			if ((a[j][j] + 1.0 == 1.0) || (a[j][j] < 0.0))
			{
				//printf("fail\n"); 
				return j;
			}
			a[j][j] = sqrt(a[j][j]);
			d = d*a[j][j];
			for (i = j + 1; i <= n - 1; i++)
			{
				u = i*n + j;
				for (k = 0; k <= j - 1; k++)
					a[i][j] = a[i][j] - a[i][k] * a[j][k];
				a[i][j] = a[i][j] / a[j][j];
			}
		}
		*det = d*d;
		for (i = 0; i <= n - 2; i++)
		for (j = i + 1; j <= n - 1; j++)
			a[i][j] = 0.0;
		return(-1);
	}
	time_t inrstart, inrend;
	time_t inrtmp1, inrtmp2 = 0;
	time_t inrstart1, inrend1;
	time_t inrtmp11, inrtmp12, inrtmp13 = 0;
	int rinv(double a[][maxn], int n)//矩阵求逆
	{
		int *is, *js, i, j, k, l, u, v;
		double d, p;
		is = (int*)malloc(n*sizeof(int));
		js = (int *)malloc(n*sizeof(int));
		inrstart = clock();
		for (k = 0; k <= n - 1; k++)
		{
			d = 0.0;
			//-------------------------------------------
			inrstart1 = clock();
			for (i = k; i <= n - 1; i++)
				for (j = k; j <= n - 1; j++)
				{
					l = i*n + j; p = fabs(a[i][j]);
					if (p > d) { d = p; is[k] = i; js[k] = j; }
				}
			inrend1 = clock();
			inrtmp11 += inrend1 - inrstart1;
			//------------------------------------------------
			if (d + 1.0 == 1.0)
			{
				free(is); free(js);
				printf("err**not inv\n");
				return(0);
			}
			//---------------------------------------
			inrstart1 = clock();
			if (is[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				u = k*n + j; v = is[k] * n + j;
				p = a[k][j]; a[k][j] = a[is[k]][j]; a[is[k]][j] = p;
			}
			if (js[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				u = i*n + k; v = i*n + js[k];
				p = a[i][k]; a[i][k] = a[i][js[k]]; a[i][js[k]] = p;
			}
			inrend1 = clock();
			inrtmp12 += inrend1 - inrstart1;
			//-----------------------------------------
			l = k*n + k;
			a[k][k] = 1.0 / a[k][k];
			for (j = 0; j <= n - 1; j++)
			if (j != k)
			{
				u = k*n + j; a[k][j] = a[k][j] * a[k][k];
			}
			//======================================================
			inrstart1 = clock();
			for (i = 0; i <= n - 1; i++)
				if (i != k)
					for (j = 0; j <= n - 1; j++)
						if (j != k) {
							u = i*n + j;
							a[i][j] = a[i][j] - a[i][k] * a[k][j];
						}
			inrend1 = clock();
			inrtmp13 += inrend1 - inrstart1;
			//---------------------------------------------
			for (i = 0; i <= n - 1; i++)
				if (i != k) {
					u = i*n + k; a[i][k] = -a[i][k] * a[k][k];
				}
		}
		inrend = clock();
		inrtmp1 += inrend - inrstart;
		inrstart = clock();
		for (k = n - 1; k >= 0; k--)
		{
			if (js[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				u = k*n + j; v = js[k] * n + j;
				p = a[k][j]; a[k][j] = a[js[k]][j]; a[js[k]][j] = p;
			}
			if (is[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				u = i*n + k; v = i*n + is[k];
				p = a[i][k]; a[i][k] = a[i][is[k]]; a[i][is[k]] = p;
			}
		}
		inrend = clock();
		inrtmp2 += inrend - inrstart;
		free(is); free(js);
		return(1);
	}

	time_t start, end;
	time_t ktmp, rtmp = 0;

	double objective(gene_model &cnt_one)//计算当前的似然函数值
	{
		
		sigma_f = cnt_one.x[0];     //parameter of the squared exponential kernel
		l = cnt_one.x[1];         //parameter of the squared exponential kernel
		sigma_n = cnt_one.x[2];        //known noise on observed data
		f1 = cnt_one.x[3];
		v = cnt_one.x[4];
		//--------------------------generate_k() running time------------------------
		start = clock();
		generate_K();
		end = clock();
		ktmp += end - start;
		
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			for (int j = 0; j < training_data_num[cnt_id]; j++){
				inv_K[i][j] = K[i][j];
			}
		}

		double det;
		int kiu = chol(inv_K, training_data_num[cnt_id], &det);  // L = inv_K

		int del_cou = 0;

		while (kiu != -1){
			printf("%d\n", kiu);
			train_data[cnt_id][kiu] = train_data[cnt_id][training_data_num[cnt_id] - 1];
			training_data_num[cnt_id]--;
			del_cou++;
			if (del_cou > 5){
				printf("THIS IS NOT GOOD\n");
			}
			generate_K();

			for (int i = 0; i < training_data_num[cnt_id]; i++){
				for (int j = 0; j < training_data_num[cnt_id]; j++){
					inv_K[i][j] = K[i][j];
				}
			}

			kiu = chol(inv_K, training_data_num[cnt_id], &det);  // L = inv_K 分解成下三角矩阵
		}

		
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			for (int j = 0; j < training_data_num[cnt_id]; j++){
				L_T_1[i][j] = L_T[i][j] = inv_K[j][i];
				L_1[i][j] = L[i][j] = inv_K[i][j];
			}
		}
		//---------------------------rinv() running time---------------------------------
		start = clock();
		rinv(L_1, training_data_num[cnt_id]);
		rinv(L_T_1, training_data_num[cnt_id]);
		end = clock();
		rtmp += end - start;
		double temp[maxn], temp2[maxn];
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			temp[i] = 0;
			for (int j = 0; j < training_data_num[cnt_id]; j++)
				temp[i] += L_1[i][j] * train_data[cnt_id][j].f;
		}

		for (int i = 0; i < training_data_num[cnt_id]; i++){
			temp2[i] = 0;
			for (int j = 0; j < training_data_num[cnt_id]; j++)
				temp2[i] += L_T_1[i][j] * temp[j];
		}
		//det = sqrt(det);
		double logv = det;
		for (int i = 0; i < training_data_num[cnt_id]; i++) logv += 0.5 * train_data[cnt_id][i].f * temp2[i];
		logv += training_data_num[cnt_id] / 2. * log(2 * PI);
		
		return logv;
	}

	int cmp(const void *a, const void *b)//比较函数
	{
		gene_model * p = (gene_model *)a;
		gene_model * q = (gene_model *)b;
		return p->f > q->f ? 1 : -1;
	}

	double SCR[POPSIZE], SF[POPSIZE];
	int s_cnt;//——————
	double uCR = 0.5, uF = 0.5;

	void initialize()//初始化种群
	{
		int i, j, k, v_id;
		uCR = 0.5, uF = 0.5;
		evals = 0;
		best_model[cnt_id].init();
		LBOUND = 0.001;	UBOUND = 90;

		archive_size = 0;
		for (i = 0; i < POPSIZE; i++){
			for (j = 0; j < MODEL_GENE_NVARS; j++){
				population[i].x[j] = randval(LBOUND, UBOUND);
			}
			population[i].f = objective(population[i]);
			if (i == 0 || population[i].f < best_model[cnt_id].f)
			{
				best_model[cnt_id] = population[i];
			}
			evals++;
			SCR[i] = randval(0, 1);
			SF[i] = randval(0, 1);
		}
		s_cnt = POPSIZE;
	}

	void adaptive_parameter()
	{
		double meanf, meanff, meancr;
		int i;
		if (s_cnt <= 0) return;
		meanf = meanff = meancr = 0;
		for (i = 0; i < s_cnt; i++){
			meanf += SF[i];
			meanff += SF[i] * SF[i];
			meancr += SCR[i];
		}
		meanf = meanff / meanf;
		meancr = meancr / s_cnt;
		uF = (1 - cC)*uF + cC*meanf;
		uCR = (1 - cC)*uCR + cC*meancr;
	}

	time_t obstart, obend, obtmp = 0;

	void production()
	{
		int i, j, k;
		int r1, r2, r3;
		double CR, F;
		adaptive_parameter();
		s_cnt = 0;
		for (i = 0; i < POPSIZE; i++){
			do{ F = cauchy(uF, 0.1); } while (F <= 0 || F >= 1);
			CR = gauss(uCR, 0.1);
			if (CR<0)CR = 0; if (CR>1)CR = 1;

			r1 = rand() % (int)(cp* POPSIZE);
			do{ r2 = rand() % POPSIZE; } while (r2 == r1);
			do{ r3 = rand() % (archive_size + POPSIZE); } while (r3 == r1 || r3 == r2);

			for (j = 0; j < MODEL_GENE_NVARS; j++){
				if (r3 < POPSIZE)
					newpopulation[i].x[j] = population[i].x[j] + F * (population[r1].x[j] - population[i].x[j]) + F * (population[r2].x[j] - population[r3].x[j]);
				else newpopulation[i].x[j] = population[i].x[j] + F * (population[r1].x[j] - population[i].x[j]) + F * (population[r2].x[j] - archives[r3 - POPSIZE].x[j]);
				if (newpopulation[i].x[j] > UBOUND) newpopulation[i].x[j] = UBOUND;
				if (newpopulation[i].x[j] < LBOUND) newpopulation[i].x[j] = LBOUND;
			}
			k = rand() % MODEL_GENE_NVARS;
			for (j = 0; j < MODEL_GENE_NVARS; j++){
				if (j == k || randval(0, 1) < CR){
					u_population[i].x[j] = newpopulation[i].x[j];
				}
				else{
					u_population[i].x[j] = population[i].x[j];
				}
			}
			//--------------------------------objective running time-------------------------
			obstart = clock();
			u_population[i].f = objective(u_population[i]);
			obend = clock();
			obtmp += obend - obstart;
			if (u_population[i].f < best_model[cnt_id].f)
			{
				best_model[cnt_id] = u_population[i];
			}
			evals++;

			if (u_population[i].f > population[i].f){
				u_population[i] = population[i];
			}
			else{
				if (archive_size < POPSIZE){
					archives[archive_size] = population[i];
					archive_size++;
				}
				else{
					j = rand() % POPSIZE;
					archives[j] = population[i];
				}
				SF[s_cnt] = F; SCR[s_cnt] = CR; s_cnt++;
			}
		}
		for (i = 0; i < POPSIZE; i++)
			population[i] = u_population[i];
	}

	int DE_for_gauss(){
		if (show_message)printf("开始训练\n");
		initialize();
		while (evals < MAXEVALS){
			production();
		}

		printf("generate_k running time = %dms\n", ktmp);
		/*printf("part11 in rinv running time = %dms\n", inrtmp11);
		printf("part12 in rinv running time = %dms\n", inrtmp12);
		printf("part13 in rinv running time = %dms\n", inrtmp13);
		printf("part1 in rinv running time = %dms\n", inrtmp1);
		printf("part2 in rinv running time = %dms\n", inrtmp2);*/
		printf("rinv running time = %dms\n", rtmp);
		printf("objective running time = %dms\n", obtmp);
		inrtmp11 = inrtmp12 = inrtmp13 = inrtmp1 = inrtmp2 = ktmp = obtmp = rtmp = 0;
		return 0;
	}

}

namespace searching{
	const int MAXEVALS = 2000;
	const int POPSIZE = 15;
	double  cp = 0.1;
	double  cC = 0.1;
	double	LBOUND[dim], UBOUND[dim];
	int		evals;
	//double	fbest;//最好的fitness函数
	//int		generation;

	int merit_Sigma;//merit 函数的参数

	const int SEA_NVARS = dim;
	NODE population[POPSIZE], newpopulation[POPSIZE], u_population[POPSIZE];
	int archive_size1;
	NODE archives[POPSIZE];

	NODE cur_p;

	double merit_svar(NODE &cnt_one){//使用标准差作为merit函数的负项
		double ans = K_ss[0][0];
		double res = 0;
		double temp[maxn], temp2[maxn];
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			temp[i] = 0;
			for (int j = 0; j < training_data_num[cnt_id]; j++){
				temp[i] += L_1[i][j] * K_s[0][j];
			}
		}
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			temp2[i] = 0;
			for (int j = 0; j < training_data_num[cnt_id]; j++){
				temp2[i] += L_T_1[i][j] * temp[j];
			}
		}
		for (int i = 0; i < training_data_num[cnt_id]; i++){
			res += K_s[0][i] * temp2[i];
		}
		ans -= res;
		return sqrt(ans);
	}

	double objective(NODE &cnt_one){//计算模型上的fitness
		K_ss[0][0] = training::ker(cnt_one, cnt_one);

		for (int ii = 0; ii < 1; ii++){
			for (int jj = 0; jj < training_data_num[cnt_id]; jj++){
				K_s[ii][jj] = training::ker(cnt_one, train_data[cnt_id][jj]);
			}
		}
		//k_s[0][i]就是k*
		int i, j, k;

		double temp[maxn], temp2[maxn];
		for (i = 0; i < training_data_num[cnt_id]; i++){
			temp[i] = 0;
			for (j = 0; j < training_data_num[cnt_id]; j++)
				temp[i] += L_1[i][j] * train_data[cnt_id][j].f;
		}

		for (i = 0; i < training_data_num[cnt_id]; i++){
			temp2[i] = 0;
			for (j = 0; j < training_data_num[cnt_id]; j++)
				temp2[i] += L_T_1[i][j] * temp[j];
		}

		double res = 0;
		for (int j = 0; j < training_data_num[cnt_id]; j++){
			res += K_s[0][j] * temp2[j];
		}

		if (merit_Sigma == 0)return res;
		return res - merit_svar(cnt_one)*merit_Sigma;
	}

	int cmp(const NODE *a, const NODE *b)//比较函数
	{
		return a->f > b->f ? 1 : -1;
	}

	double SCR[POPSIZE], SF[POPSIZE];
	int s_cnt1;//——————
	double uCR = 0.5, uF = 0.5;

	void initialize()//初始化种群
	{
		int i, j, k, v_id;
		uCR = 0.5, uF = 0.5;
		evals = 0;
		//cnt_round_p[cnt_id].init();
		cur_p.init();
		for (int i = 0; i < dim; i++){
			LBOUND[i] = solution_space[cnt_id][i][0];
			UBOUND[i] = solution_space[cnt_id][i][1];
		}

		searching::archive_size1 = 0;
		for (i = 0; i < POPSIZE; i++){
			for (j = 0; j < dim; j++){
				population[i].x[j] = randval(LBOUND[j], UBOUND[j]);
			}
			population[i].f = objective(population[i]);
			if (population[i].f < cur_p.f)
			{
				cur_p = population[i];
			}
			evals++;
			SCR[i] = randval(0, 1);
			SF[i] = randval(0, 1);
		}
		searching::s_cnt1 = POPSIZE;
	}

	void adaptive_parameter()
	{
		double meanf, meanff, meancr;
		int i;
		if (searching::s_cnt1 <= 0) return;
		meanf = meanff = meancr = 0;
		for (i = 0; i < searching::s_cnt1; i++){
			meanf += SF[i];
			meanff += SF[i] * SF[i];
			meancr += SCR[i];
		}
		meanf = meanff / meanf;
		meancr = meancr / searching::s_cnt1;
		uF = (1 - cC)*uF + cC*meanf;
		uCR = (1 - cC)*uCR + cC*meancr;
	}

	void production()
	{
		int i, j, k;
		int r1, r2, r3;
		double CR, F;
		adaptive_parameter();
		searching::s_cnt1 = 0;
		for (i = 0; i < POPSIZE; i++){
			do{ F = cauchy(uF, 0.1); } while (F <= 0 || F >= 1);
			CR = gauss(uCR, 0.1);
			if (CR<0)CR = 0; if (CR>1)CR = 1;

			r1 = rand() % (int)(cp* POPSIZE);
			do{ r2 = rand() % POPSIZE; } while (r2 == r1);
			do{ r3 = rand() % (searching::archive_size1 + POPSIZE); } while (r3 == r1 || r3 == r2);

			for (j = 0; j < dim; j++){
				if (r3 < POPSIZE)
					newpopulation[i].x[j] = population[i].x[j] + F * (population[r1].x[j] - population[i].x[j]) + F * (population[r2].x[j] - population[r3].x[j]);
				else newpopulation[i].x[j] = population[i].x[j] + F * (population[r1].x[j] - population[i].x[j]) + F * (population[r2].x[j] - archives[r3 - POPSIZE].x[j]);
				if (newpopulation[i].x[j] > UBOUND[j]) newpopulation[i].x[j] = UBOUND[j];
				//if (newpopulation[i].x[j] > UBOUND[j]||newpopulation[i].x[j] < LBOUND[j]) newpopulation[i].x[j] = randval(LBOUND[j],UBOUND[j]);
				if (newpopulation[i].x[j] < LBOUND[j]) newpopulation[i].x[j] = LBOUND[j];
			}
			k = rand() % dim;
			for (j = 0; j < dim; j++){
				if (j == k || randval(0, 1) < CR){
					u_population[i].x[j] = newpopulation[i].x[j];
				}
				else{
					u_population[i].x[j] = population[i].x[j];
				}
			}
			u_population[i].f = objective(u_population[i]);
			if (u_population[i].f < cur_p.f)
			{
				cur_p = u_population[i];
			}
			evals++;

			if (u_population[i].f > population[i].f){
				u_population[i] = population[i];
			}
			else{
				if (searching::archive_size1 < POPSIZE){
					archives[searching::archive_size1] = population[i];
					searching::archive_size1++;
				}
				else{
					j = rand() % POPSIZE;
					archives[j] = population[i];
				}
				SF[searching::s_cnt1] = F; SCR[searching::s_cnt1] = CR; searching::s_cnt1++;
			}
		}
		for (i = 0; i < POPSIZE; i++)
			population[i] = u_population[i];
	}

	void DE(){//每次de找到当前模型在效益函数下的最小值点
		double cc = 0x3f3f3f3f;
		initialize();
		while (evals < MAXEVALS){
			production();
			//qsort(population, POPSIZE, sizeof(population[0]), cmp);
			//printf("%d\t%g\n", generation, population[0].f);
		}
	}


	double det;
	void init(gene_model cnt_one){//用最优参数去初始化高斯过程所需要的东西
		sigma_f = cnt_one.x[0];
		l = cnt_one.x[1];
		sigma_n = cnt_one.x[2];
		f1 = cnt_one.x[3];
		v = cnt_one.x[4];
		training::generate_K();

		int i, j, k;
		for (i = 0; i < training_data_num[cnt_id]; i++){
			for (j = 0; j < training_data_num[cnt_id]; j++){
				inv_K[i][j] = K[i][j];
			}
		}

		i = training::chol(inv_K, training_data_num[cnt_id], &det);  // L = inv_K;

		for (i = 0; i < training_data_num[cnt_id]; i++){
			for (j = 0; j < training_data_num[cnt_id]; j++){
				L_T_1[i][j] = L_T[i][j] = inv_K[j][i];
				L_1[i][j] = L[i][j] = inv_K[i][j];
			}
		}

		training::rinv(L_1, training_data_num[cnt_id]);
		training::rinv(L_T_1, training_data_num[cnt_id]);

	}

	int sgv[4] = { 0, 1, 2, 4 };

	void DE_find_min(){
		if (show_message)printf("Searching...for function —————%d————\n", cnt_id);
		searching::init(best_model[cnt_id]);//初始化训练好的高斯过程模型======================check

		cnt_round_p[cnt_id].f = inf;

		for (int kk = 0; kk < 4; kk++){
			merit_Sigma = sgv[kk];
			searching::DE();//搜索模型上的最小值=========================check

			if (show_message){
				printf("sigma=%d\tMin_Point:", merit_Sigma);
				printf("——fitness on model = %.10lf", cur_p.f);
			}
			cur_p.f = fitness_function(mapping(cur_p).x);
			if (show_message)printf("——real fitess = %.10lf\n", cur_p.f);

			tot_data[cnt_id][tot_data_num[cnt_id]++] = cur_p;
			recent_data[cnt_id][rencent_point_num[cnt_id]++] = cur_p;

			if (cur_p.f < cnt_round_p[cnt_id].f){//更新此轮的最优值
				cnt_round_p[cnt_id] = cur_p;
			}
			if (cur_p.f < real_best_p[cnt_id].f){//更新最优点
				real_best_p[cnt_id] = cur_p;
			}

		}

	}

}

void init(int cnt_id){

	for (int i = 0; i < dim; i++){
		solution_space[cnt_id][i][0] = global_lbound;
		solution_space[cnt_id][i][1] = global_rbound;
	}

	tot_data_num[cnt_id] = Nc;
	rencent_point_num[cnt_id] = 0;
	for (int i = 0; i < tot_data_num[cnt_id]; i++){
		for (int j = 0; j < dim; j++){
			tot_data[cnt_id][i].x[j] = randval(solution_space[cnt_id][j][0], solution_space[cnt_id][j][1]);
		}

		tot_data[cnt_id][i].f = fitness_function(mapping(tot_data[cnt_id][i]).x, cnt_id);
		if (tot_data[cnt_id][i].f < real_best_p[cnt_id].f){
			real_best_p[cnt_id] = tot_data[cnt_id][i];
		}

	}
}

bool cmp1(NODE &a, NODE &b){
	return dis(a, real_best_p[cnt_id]) < dis(b, real_best_p[cnt_id]);
}

int same_point(NODE &a, NODE &b){
	int cc = 0;
	for (int i = 0; i < dim; i++){
		if (a.x[i] == b.x[i])cc++;
	}
	return cc;
}

bool choose_point(){

	training_data_num[cnt_id] = 0;

	sort(tot_data[cnt_id], tot_data[cnt_id] + tot_data_num[cnt_id], cmp1);
	for (int i = 0; i < tot_data_num[cnt_id]; i++){
		train_data[cnt_id][training_data_num[cnt_id]++] = tot_data[cnt_id][i];
		if (training_data_num[cnt_id] == Nc)break;
	}

	for (int i = 0; i < dim; i++){
		solution_space[cnt_id][i][0] = train_data[cnt_id][0].x[i];
		solution_space[cnt_id][i][1] = train_data[cnt_id][0].x[i];
	}

	for (int i = 1; i < training_data_num[cnt_id]; i++){
		for (int j = 0; j < dim; j++){
			solution_space[cnt_id][j][0] = min(solution_space[cnt_id][j][0], train_data[cnt_id][i].x[j]);
			solution_space[cnt_id][j][1] = max(solution_space[cnt_id][j][1], train_data[cnt_id][i].x[j]);
		}
	}

	bool convengence = true;
	for (int i = 0; i < dim; i++){
		double d = solution_space[cnt_id][i][1] - solution_space[cnt_id][i][0];
		if (d >= 1e-4)convengence = false;
		d /= 2;
		solution_space[cnt_id][i][0] = real_best_p[cnt_id].x[i] - d;
		solution_space[cnt_id][i][1] = real_best_p[cnt_id].x[i] + d;
	}
	for (int j = 0; j < dim; j++){
		if (solution_space[cnt_id][j][0] < global_lbound)solution_space[cnt_id][j][0] = global_lbound;
		if (solution_space[cnt_id][j][1] > global_rbound)solution_space[cnt_id][j][1] = global_rbound;
	}


	int cou = 0;
	for (int i = rencent_point_num[cnt_id] - 1; i >= 0; i--){
		bool f = 1;
		for (int j = 0; j < Nc; j++){
			if (recent_data[cnt_id][i] == train_data[cnt_id][j]){ f = 0; break; }
		}
		if (f == 0)continue;
		train_data[cnt_id][training_data_num[cnt_id]++] = recent_data[cnt_id][i];
		cou++;
		if (cou == Nr)break;
	}


	return false;//强制设置为未收敛

}

double l_K[maxn][maxn], r_K[maxn][maxn];//covariance 
double l_N[maxn][maxn], r_N[maxn][maxn];//两个矩阵的逆
const double dkl_inf = 1e9;

void Gauss_surrogate(){
	for (int i = 0; i < func_num; i++){
		real_best_p[i].f = inf;
	}
	for (cnt_id = 0; cnt_id < func_num; cnt_id++){
		init(cnt_id);
	}
	while (fit_time[0]<MAX_FIT_TIME){
		time_t start, end;
		start = clock();
		for (cnt_id = 0; cnt_id < func_num; cnt_id++){
			if (end_f[cnt_id])continue;
			if (choose_point() == true){
				end_f[cnt_id] = 1;
				continue;
			}
			time_t a, b;
			a = clock();
			training::DE_for_gauss();
			b = clock();
			printf("DE_for_gauss running time = %dms\n", b-a);
			a = clock();
			searching::DE_find_min();
			b = clock();
			printf("DE_find_min running time = %dms\n", b - a);
		}
	
	
		if (1){
			printf("\n====================================\n");
			for (int k = 0; k < func_num; k++){
				if (end_f[k])printf("end==");
				printf("%d -- best fitness=%.10lf\n", k, real_best_p[k].f);
			}
			printf("\n====================================\n");
		}
		end = clock();
		printf("all running time = %dms\n", end - start);
		fit_time[0]++;
	}

}

int main(){
	//srand(time(0));
	time_t s, f;
	Gauss_surrogate();
	system("pause");
	return 0;
}