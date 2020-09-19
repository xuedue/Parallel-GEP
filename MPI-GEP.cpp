#define _CRT_SECURE_NO_DEPRECATE
#include "mpi.h"

#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "math.h"
#include "string.h"
#include <algorithm>
using namespace std;

#define H   10			//head length of the main program
#define T	(H+1)		//tail length of the main program   (the maximum arities of all functions is 2)
#define GSIZE 2			//number of ADFs
#define GH	3			//head length of ADFs
#define GT		(GH+1)	//tail length of ADFs
#define GNVARS	(GH+GT)	
#define NVARS	(H+T + GSIZE *(GH+GT))	//chromosome length

#define POPSIZE	50					//population size
#define MAX_TERMINAL_NUM	10		//maximun terminal number
int		L_terminal = 10000;			//start value of terminal symbol
int		L_input = 20000;			//start value of input symbol
int		base_function_num = 8;		//{and, sub, mul, div,  sin, cos, exp, log}
int		generation = 0;					//number of generations
int		terminal_num = 1;			//current number of terminals
int		function_num = (base_function_num + GSIZE);			//total function numbers including the ADFs
bool	variable_value[MAX_TERMINAL_NUM];					//input variable values
int		gene_type_flag[NVARS];								//the type of each bit in the chromosome

typedef struct
{
	int gene[NVARS];
	double f;
}CHROMOSOME;
CHROMOSOME population[POPSIZE+1], newpopulation[POPSIZE],subpopulation[POPSIZE],tempbestpopulation;

//========for stochastical analysis ===========================
#define MAXEVALS 1000000
#define MAXGENS	20000
double	fbest;
double  tempfbest;
int		evals;
//============= nodes and tree for computing the fitness of individuals ==============================
#define		MAX_SIBLING	20				//the maximum sibling for each node
#define		LINK_LENGTH	(NVARS * 20)	//add enough to save all necessary node.
struct LINK_COMP
{
	int value;							// node label
	int sibling_num;			
	struct LINK_COMP *siblings[MAX_SIBLING];
};
struct LINK_COMP *link_root, link_comp[LINK_LENGTH];		//the whole expression tree
struct LINK_COMP *sub_root[GSIZE], sub_comp[GSIZE][GNVARS]; //the sub expression tree 

//=============== parameters for symbolic regression problem ======================================================
int		function	=	0;		//current problem index
int		job	=	0;				//EA run index 
#define MAXINPUTS		200	//maximum input-output pairs of each problem 
#define	MAX_VARIABLES	3
int		input_num;
double	current_value[MAXINPUTS];
double	training_inputs[MAXINPUTS][MAX_VARIABLES];
double	training_outputs[MAXINPUTS];
int		training_cases;

//for sub expression trees
double sub_sibling_value[MAX_SIBLING][MAXINPUTS];
double sub_current_value[MAXINPUTS];

//return a uniform random nubmer within [a,b)
double randval(double a, double b)
{
	return a + (b - a) * rand() /(double) RAND_MAX;
}


void read_data(int run)
{
	int i, j;
	FILE *f;
	char name[200];
	i = (int) (run % 10);
	sprintf(name, "F:\\guna\\spark\\f1.txt");
	//printf(name);
	//sprintf(name, "F:\\guna\\GA\\ten_fold_cross_validation\\F%d_%d_training_data.txt",function, i);
	f = fopen(name,"r");
	int row_num, col_num;
	fscanf(f, "%d\t%d\n", &row_num, &col_num);
	input_num = training_cases = row_num;
	for(i = 0; i < row_num; i++){
		for(j = 0; j < col_num; j++){
			fscanf(f,"%lf\t", &training_inputs[i][j]);
		}
		fscanf(f,"%lf\n", &training_outputs[i]);
	}
	/*printf("%d\t%d\n",row_num,col_num);
	for(i = 0; i < row_num; i++){
		for(j = 0; j < col_num; j++){
			printf("%lf\t", &training_inputs[i][j]);
		}
		printf("%lf\n", &training_outputs[i]);
	}*/
}

//compute the sub-tree
void compute_sub_rule(const struct LINK_COMP * node)
{
	int i;
	double *t2;
	t2 = (double*)malloc(MAXINPUTS * sizeof(double));
	if(node->value >= L_input){
		// If the node is an input then read data from the input vector, i.e., sub_sibling_value[...];
		for(i = 0; i < input_num; i++) 	sub_current_value[i] = sub_sibling_value[node->value - L_input][i];
		
	}else{
		// First compute the left child of the node.
		double t1[MAXINPUTS];
		compute_sub_rule(node->siblings[0]);

		for(i = 0; i < input_num; i++) t1[i] = sub_current_value[i];
		//then compute the right child of the node if the node contain right child
		if(node->value < 4){  // note that the first 4 functions have 2 children
			compute_sub_rule(node->siblings[1]);
			for(i = 0; i < input_num; i++) t2[i] = sub_current_value[i];
		}
		switch(node->value){
		case 0: //+ 			
				for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] + t2[i]; break;
		case 1: //-
				for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] - t2[i]; break;
		case 2: //*
				for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] * t2[i]; break;
		case 3: // /
				for(i = 0; i < input_num; i++){ if(fabs(t2[i]) <  1e-20) sub_current_value[i] = 0;else sub_current_value[i] = t1[i] / t2[i];} break;
		case 4: //sin
				for(i = 0; i < input_num; i++){ sub_current_value[i] = sin(t1[i]); } break;
		case 5: //cos
				for(i = 0; i < input_num; i++){ sub_current_value[i] = cos(t1[i]); } break;
		case 6: //exp
				for(i = 0; i < input_num; i++){ if(t1[i] < 20) sub_current_value[i] = exp(t1[i]); else sub_current_value[i] = exp(20.); } break;
		case 7: //log
				for(i = 0; i < input_num; i++){ if(fabs(t1[i]) <  1e-20) sub_current_value[i] = 0; else sub_current_value[i] = log(fabs(t1[i])); } break;
		default: printf("unknow function\n");
		}
	}
	free(t2);
}

//Compute the entire solution tree.
void compute_rule(const struct LINK_COMP * node)
{
	int i;
	double *t2;
	t2 = (double*)malloc(MAXINPUTS * sizeof(double));
	if(node->value >= L_terminal){
		for(i = 0; i < input_num; i++)
			current_value[i] = training_inputs[i][node->value - L_terminal];		
	}else{
		double t1[MAXINPUTS];
		compute_rule(node->siblings[0]);
		for(i = 0; i < input_num; i++) t1[i] = current_value[i];
		if(node->value < 4 || node->value >= base_function_num){
			compute_rule(node->siblings[1]);
			for(i = 0; i < input_num; i++) t2[i] = current_value[i];
		}
		switch(node->value){
		case 0: //+ 			
				for(i = 0; i < input_num; i++) current_value[i] = t1[i] + t2[i]; break;
		case 1: //-
				for(i = 0; i < input_num; i++) current_value[i] = t1[i] - t2[i]; break;
		case 2: //*
				for(i = 0; i < input_num; i++) current_value[i] = t1[i] * t2[i]; break;
		case 3: // /
				for(i = 0; i < input_num; i++){ if(fabs(t2[i]) <  1e-20) current_value[i] = 0;else current_value[i] = t1[i] / t2[i];} break;
		case 4: //sin
				for(i = 0; i < input_num; i++){ current_value[i] = sin(t1[i]); } break;
		case 5: //cos
				for(i = 0; i < input_num; i++){ current_value[i] = cos(t1[i]); } break;
		case 6: //exp
				for(i = 0; i < input_num; i++){ if(t1[i] < 20) current_value[i] = exp(t1[i]); else current_value[i] = exp(20.); } break;
		case 7: //log
				for(i = 0; i < input_num; i++){ if(fabs(t1[i]) <  1e-20) current_value[i] = 0; else current_value[i] = log(fabs(t1[i])); } break;

		default: //GI
				for(i = 0; i < input_num; i++){ sub_sibling_value[0][i] = t1[i]; sub_sibling_value[1][i] = t2[i];}				
				compute_sub_rule(sub_root[node->value - 8]);
				for(i = 0; i < input_num; i++){ current_value[i] = sub_current_value[i];}
				break;
		}
	}
	free(t2);
}



//Decode the chromosome, build the main expression tree, including sub-expression trees.
void decode_gene( CHROMOSOME * p)
{	
	int op = -1, i = 0, k = 0, j;
	for(i = 0; i < NVARS; i++){			
		link_comp[i].value = p->gene[i];
		for(j = 0; j < MAX_SIBLING; j++)
			link_comp[i].siblings[j] = NULL;
	}

	op = -1, i = 1;
	link_root = &link_comp[0];
	if(link_root->value < function_num){
		do{ 
			//find an op type item
			do{op++; if(op >= i)break;}while(link_comp[op].value >= L_terminal);
			if(op >= i) break;
			//set its left and right;
			if(link_comp[op].value < L_terminal){
				if(i >= H+T){break;}
				link_comp[op].siblings[0] = &link_comp[i];				
				i++;
				if(link_comp[op].value < 4 || link_comp[op].value >= base_function_num){
					if(i >= H+T){ break;}
					link_comp[op].siblings[1] = &link_comp[i];
					i++;
				}
			}
		}while(true);

		if(op < i  && i >= H+T){ 			
			printf("\nERROR RULE111"); 
			getchar();
		}
	}else{
		//printf("terminate");
	}

	//build sub expression trees of the individual
	for(int g = 0; g < GSIZE; g++){
		k = H+T + g *GNVARS;	// the starting position of the ADF.	
		for(i = 0; i < GNVARS; i++){
			sub_comp[g][i].value =  p->gene[k + i];
			for(j = 0; j < MAX_SIBLING; j++)
				sub_comp[g][i].siblings[j] = NULL;
		}
		op = -1, i = 1;
		sub_root[g] = &sub_comp[g][0];
		if(sub_root[g]->value < L_terminal){  // note that L_input > L_terminal;
			do{ //find an op type item
				do{op++; if(op >= i)break;}while(sub_comp[g][op].value >= L_terminal);
				if(op >= i) break;
				//set its left and right;
				if(sub_comp[g][op].value < base_function_num){
					if(i >= GH+GT-1){ break;}
					sub_comp[g][op].siblings[0] = &sub_comp[g][i];				
					i++;
					if(sub_comp[g][op].value < 4){
						sub_comp[g][op].siblings[1] = &sub_comp[g][i];
						i++;
					}
				}
			}while(true);
			if(op < i  && i >= GH+GT - 1){ printf("SUB ERROR RULE111"); getchar();}
		}else{ 
			//printf("SUB terminate");
		}
	}
}

void objective(CHROMOSOME * p)
{
	p->f = 1e10;
	decode_gene( p);	
	compute_rule(link_root);	
	
	double v = 0;
	for(int j = 0; j < input_num; j++){
		v += (training_outputs[j] - current_value[j])*(training_outputs[j] - current_value[j]);	
	}
	v = sqrt(v/input_num);
	//printf("objective v=%lf\n", v);
	if(v < 1e-4) v = 0;

	p->f = v;
	if(v < fbest){
		fbest = v;
	}
	evals ++;
	
}



//================================================================================
//randomly set the value of the I-th bit of an individual, x points to this bit.
//There are only four possibles: 0: the main head; 1: the main tail; 2: the sub head; 3: the sub tail;
void rand_set_value(int I, int*x)
{	
	switch(gene_type_flag[I]){
	case 0: 
		if(randval(0, 1) < 1./3) *x = rand()%(base_function_num);		// note that function_num = base_function_num + GSIZE;
		else if(randval(0,1) < 0.5) *x = base_function_num + rand()%(GSIZE);
		else *x = L_terminal + rand() % (terminal_num);
		break;
	case 1: *x = L_terminal + rand() % (terminal_num);
		break;
	case 2: if(rand()%2==0)	*x = rand()%(base_function_num);
		else *x = L_input + rand()%(2); 
		break; 
	case 3:  *x = L_input + rand()%(2);break;
	default: printf("fds");
	}
}

//===============================probability of components ============================================================ 
double	FQ;										//in the main heads of population, the proportion of bits being function symbol
#define MAXIMUM_ELEMENTS	100					//MAXIMUM_ELEMENTS > function_num && MAXIMUM_ELEMENTS > terminal_num
double	function_freq[MAXIMUM_ELEMENTS];						//in the main parts of population, the frequency of each function symbol
double	terminal_freq[MAXIMUM_ELEMENTS];						//in the main parts of population, the frequency of each terminal symbol
double	terminal_probability[MAXIMUM_ELEMENTS];				//store the selection probability of each terminal
double	function_probability[MAXIMUM_ELEMENTS];				//store the selection probability of each function
void update_probability()
{
	double sum = 0;
	int i,j;
	//in the main head of population, the proportion of bits being function symbol
	FQ = 0;
	int	CC = 0;
	for(i = 0; i < POPSIZE; i++){
		for(j = 0; j < H; j++){
			if(population[i].gene[j] < L_terminal) FQ ++;
			else if(population[i].gene[j] >= L_terminal) CC++;
		}
	}
	FQ = FQ / (double) (POPSIZE * H);

	bool print_flag = false;
	
	//now compute the frequency of each symbol in the main parts of the current population.
	for(i = 0; i < MAXIMUM_ELEMENTS; i++){
		function_freq[i] = 1;	//initialize a very small value.
		terminal_freq[i] = 1;
		
	}

	for(i = 0; i < POPSIZE; i++){
		for(j = 0; j < H+T; j++){  //only consider main parts
			if(population[i].gene[j] < L_terminal){
				
				function_freq[population[i].gene[j]]++;
			}else
				terminal_freq[population[i].gene[j] - L_terminal] ++;
		}
	}
	
	sum = 0;
	for(i = 0; i < function_num; i++){
		sum +=function_freq[i];
	}
	function_probability[0] =  function_freq[0] / sum;
	for(i = 1; i < function_num; i++){
		function_probability[i] = function_freq[i] / sum + function_probability[i - 1];		
	}

	sum = 0;
	for(i = 0; i < terminal_num; i++){
		sum += terminal_freq[i];
		terminal_probability[i] = terminal_freq[i];
	}
	terminal_probability[0] =  terminal_probability[0] / sum;
	for(i = 1; i < terminal_num; i++){
		terminal_probability[i] = terminal_probability[i] / sum + terminal_probability[i - 1];	
	}
}

//choose a terminal according to its frequence.
int choose_a_terminal()
{
	int i;
	double p = randval(0,1);
	for(i = 0; i < terminal_num - 1; i++){
		if(p < terminal_probability[i])
			break;
	}
	return L_terminal+i;
}

//choose a function according to its frequence.
int choose_a_function()
{	
	int i;
	double p = randval(0,1);
	for(i = 0; i < function_num - 1; i++){
		if(p < function_probability[i])
			break;
	}
	return i;
}

//bially set value of bits. 
void biasly_set_value(int I, int*x)
{	
	//here we only consder the main parts, while the sub-gene part are also randomly setting, so as to import population diversity.
	switch(gene_type_flag[I]){
	case 0: 
		if(randval(0, 1) < FQ) *x = choose_a_function();
		else *x = choose_a_terminal();
		break;
	case 1: *x = choose_a_terminal(); break;
	case 2: 
		if(rand()%2==0) *x = rand()%(base_function_num);
		else *x = L_input + rand()%(2); 
		break;
	case 3: *x = L_input + rand()%(2);break;
	default: printf("fds");
	}
}

void initialize()
{
	int i, j, k;
	int ibest = 0;
	evals = 0;
	fbest = 1e10;
	//firstly set the type of each bit.
	for(i = 0; i < NVARS; i++){
		if(i < H)  gene_type_flag[i] = 0;
		else if(i< H + T)  gene_type_flag[i] = 1;
		else{
			j = i - H - T;
			if(j%(GH+GT) < GH) gene_type_flag[i] = 2;
			else gene_type_flag[i] = 3;
		}	
	}

	for(i = 0; i < POPSIZE; i++){
		for(k = 0; k < NVARS; k++){
			rand_set_value(k, &population[i].gene[k]);	
		}
		objective(&population[i]);	
		if(population[i].f < population[ibest].f) ibest = i;	
	}
	population[POPSIZE] = population[ibest];
}

int total_process;
int mypop;

void production(int process_id)
{
	int i, j, k, r1, r2;
	int index;
	double CR, F;
	double change_vector[NVARS];	
	update_probability();
	int startindex, endindex;
	startindex = (process_id - 1)*mypop;
	if (process_id != total_process - 1)
		endindex = startindex + mypop;
	else endindex = POPSIZE;
	index = 0;
	for(i = startindex; i < endindex; i++,index++){
		F = randval(0, 1);
		CR = randval(0,1);
		do{ r1 = rand() % (POPSIZE);}while(r1 == i);
		do{r2 = rand() % POPSIZE;}while(r2 == r1 || r2 == i);	
		k = rand() % NVARS;
		for(j = 0; j < NVARS; j++){
			if(randval(0,1) < CR || k == j){			
				double dd1 = 0;
				if(((int)population[POPSIZE].gene[j]) != ((int) population[i].gene[j])) dd1 = 1;
				double dd2 = 0;
				if(((int)population[r1].gene[j]) != ((int) population[r2].gene[j])) dd2 = 1;
				change_vector[j] = F * dd1 + F * dd2 - (F * dd1 * F * dd2);
				if(randval(0,1) < change_vector[j]){
					biasly_set_value(j, &subpopulation[index].gene[j]);
				}else{
					subpopulation[index].gene[j] =  population[i].gene[j];
				}
			}else{
				change_vector[j] = 0;
				subpopulation[index].gene[j] = population[i].gene[j];
			}
		}
		objective(&subpopulation[index]);
		if(subpopulation[index].f < population[i].f){
			population[i] = subpopulation[index];
			if(population[i].f < population[POPSIZE].f){
				population[POPSIZE] = population[i];
			}		
		}
		subpopulation[index] = population[i];
	}
}

int main(int argc, char *argv[])
{	
	MPI_Init(&argc, &argv);
	int process_id;
	int finalindex;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &total_process);
	srand(time(NULL));
	function = 0;
	job = 0;
	time_t start, finish;
	read_data(job);
	initialize();
	//SLGEP(argc,argv);	
	//--------------SLGEP-----------------------
	printf("我是线程%d，现在是%d代,总线程=%d\n", process_id, generation,total_process);
	MPI_Status status;
	//read_data(process_id);
	//自定义数据类型
	MPI_Datatype mystruct;
	int blocklens[2];
	MPI_Aint indices[2];
	MPI_Datatype old_types[6];
	//各块中数据个数
	blocklens[0] = NVARS;
	blocklens[1] = 1;
	//数据类型
	old_types[0] = MPI_INT;
	old_types[1] = MPI_DOUBLE;
	//求地址和相对偏移
	MPI_Get_address(&population->gene, &indices[0]);
	MPI_Get_address(&population->f, &indices[1]);
	indices[1] = indices[1] - indices[0];
	indices[0] = 0;
	MPI_Type_create_struct(2, blocklens, indices, old_types, &mystruct);
	MPI_Type_commit(&mystruct);

	mypop = POPSIZE / (total_process - 1);
	finalindex = (total_process - 2)*mypop;

	if (process_id == 0)
		start = clock();

	while (generation < 3000) {
		if (process_id == 0) {
			//init初始化
			for (int i = 1; i < total_process; i++) {
				MPI_Send(&fbest, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(population, POPSIZE + 1, mystruct, i, 1, MPI_COMM_WORLD);
			}
			/*if (population[POPSIZE].f < 1e-4) {
				printf("我是线程%d，第%d代收敛完成\n", process_id, generation);
				break;
			}*/
			//发送，分配
			for (int i = 1; i < total_process - 1; i++) {
				for (int j = 0; j < mypop; j++)
					subpopulation[j] = population[mypop*(i - 1) + j];
				MPI_Send(subpopulation, mypop, mystruct, i, 2, MPI_COMM_WORLD);
			}
			for (int j = 0; j < POPSIZE - finalindex; j++)
				subpopulation[j] = population[finalindex + j];
			MPI_Send(subpopulation, (POPSIZE - finalindex), mystruct, total_process - 1, 2, MPI_COMM_WORLD);
			//收回
			for (int i = 1; i < total_process - 1; i++) {
				MPI_Recv(subpopulation, mypop, mystruct, i, 3, MPI_COMM_WORLD, &status);
				for (int j = 0; j < mypop; j++)
					population[mypop*(i - 1) + j] = subpopulation[j];
				MPI_Recv(&tempfbest, 1, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
				if (tempfbest < fbest) fbest = tempfbest;
				MPI_Recv(&tempbestpopulation, 1, mystruct, i, 5, MPI_COMM_WORLD, &status);
				if (tempbestpopulation.f < population[POPSIZE].f) population[POPSIZE] = tempbestpopulation;
			}
			MPI_Recv(subpopulation, (POPSIZE - finalindex), mystruct, total_process - 1, 3, MPI_COMM_WORLD, &status);
			for (int j = 0; j < POPSIZE - finalindex; j++)
				population[finalindex + j] = subpopulation[j];
			MPI_Recv(&tempfbest, 1, MPI_DOUBLE, total_process - 1, 4, MPI_COMM_WORLD, &status);
			if (tempfbest < fbest) fbest = tempfbest;
			MPI_Recv(&tempbestpopulation, 1, mystruct, total_process - 1, 5, MPI_COMM_WORLD, &status);
			if (tempbestpopulation.f < population[POPSIZE].f) population[POPSIZE] = tempbestpopulation;
		}
		else {
			if (process_id != total_process - 1) {
				//receive init
				MPI_Recv(&fbest, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(population, POPSIZE + 1, mystruct, 0, 1, MPI_COMM_WORLD, &status);
				/*if (population[POPSIZE].f < 1e-4) {
					printf("我是线程%d，第%d代收敛完成\n", process_id, generation);
					break;
				}*/
				//--------
				MPI_Recv(subpopulation, mypop, mystruct, 0, 2, MPI_COMM_WORLD, &status);
				production(process_id);
				MPI_Send(subpopulation, mypop, mystruct, 0, 3, MPI_COMM_WORLD);
				MPI_Send(&fbest, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
				MPI_Send(&population[POPSIZE], 1, mystruct, 0, 5, MPI_COMM_WORLD);
			}
			else {
				//receive init
				MPI_Recv(&fbest, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(population, POPSIZE + 1, mystruct, 0, 1, MPI_COMM_WORLD, &status);
				/*if (population[POPSIZE].f < 1e-4) {
					printf("我是线程%d，第%d代收敛完成\n", process_id, generation);
					break;
				}*/
				//--------------
				MPI_Recv(subpopulation, (POPSIZE - finalindex), mystruct, 0, 2, MPI_COMM_WORLD, &status);
				production(process_id);
				MPI_Send(subpopulation, (POPSIZE - finalindex), mystruct, 0, 3, MPI_COMM_WORLD);
				MPI_Send(&fbest, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
				MPI_Send(&population[POPSIZE], 1, mystruct, 0, 5, MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		/*if (population[POPSIZE].f < 1e-4) {
			printf("第%d代收敛完成\n", generation);
			break;
		}*/
		if (process_id == 0 && generation % 100 == 0)
			printf("%d\t%d\t%d\t%g\n", function, job, generation, fbest);
		generation++;
	}
	if (process_id == 0) {
		finish = clock();
		printf("running time = %dms\n", finish - start);
	}
	MPI_Type_free(&mystruct);
	MPI_Finalize();
	return 0;
}
