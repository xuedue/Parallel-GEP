package sparktest;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;  
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.*;



public class HelloSpark implements Serializable{
	
	static JavaSparkContext sc;
	
	static int H=10;
	static int T=H+1;
	static int GSIZE=2;
	static int GH=3;
	static int GT=GH+1;
	static int GNVARS=GH+GT;
	static int NVARS=(H+T+GSIZE*(GH+GT));
	
	static int POPSIZE=50;
	final int MAX_TERMINAL_NUM=10;
	
	static int		L_terminal = 10000;			//start value of terminal symbol
	static int		L_input = 20000;			//start value of input symbol
	static int		base_function_num = 8;		//{and, sub, mul, div,  sin, cos, exp, log}操作
	static int		generation = 1;					//number of generations代数
	static int		terminal_num = 2;			//current number of terminals
	static int		function_num = (base_function_num + GSIZE);			//total function numbers including the ADFs总操作数
	boolean[]	variable_value=new boolean[MAX_TERMINAL_NUM];					//input variable values
	static int	[]	gene_type_flag=new int[NVARS];	
	
	static int row_num;
	static int col_num = 0;
	static int ttime1 = 0;
	static int ttime2 = 0;
	static int ttime3 = 0;
	
	public static class CHROMOSOME implements Serializable
	{
		 int[] gene=new int[NVARS];
		 double f;
		 int index;
		 public CHROMOSOME() {
			 
		 }
	}
	 static CHROMOSOME[] population=new CHROMOSOME[POPSIZE+1];
	 static CHROMOSOME[] newpopulation=new CHROMOSOME[POPSIZE];
	
	static int MAXEVALS=1000000;
	static int MAXGENS=20000;
	static double fbest;
	static int evals;
	
	static int MAX_SIBLING=20;
	static int LINK_LENGTH=(NVARS*20);
	
	public static class LINK_COMP implements Serializable{
		int value;
		int sibling_num;
		LINK_COMP [] siblings=new LINK_COMP[MAX_SIBLING];
	}
	static LINK_COMP link_root = new LINK_COMP();
	static LINK_COMP link_root1 = new LINK_COMP();
	static LINK_COMP[] link_comp=new LINK_COMP[LINK_LENGTH];
	static LINK_COMP[] sub_root=new LINK_COMP[GSIZE];
	static LINK_COMP sub_comp[][]=new LINK_COMP[GSIZE][GNVARS];
	
	static void assign(CHROMOSOME source, CHROMOSOME destination) {
		int i;
		for (i = 0; i < NVARS; i++) {
			destination.gene[i] = source.gene[i];
		}
		destination.index = source.index;
		destination.f = source.f;
	}
	static int function=0;
	static int job=0;
	static int MAXINPUTS=1000;
	static int MAX_VARIABLES=3;
	static int input_num;
	static double[] current_value=new double[MAXINPUTS];
	static double training_inputs[][]=new double[MAXINPUTS][MAX_VARIABLES];
	static double[] training_outputs=new double[MAXINPUTS];
	static int training_cases;
	
	static double sub_sibling_value[][]=new double[MAX_SIBLING][MAXINPUTS];
	static double[] sub_current_value=new double[MAXINPUTS];
	
	static Random rand = new Random();
	
	static double randval(double a, double b)
	{	 
		return a + (b - a) * rand.nextDouble();
	}
	
	static void read_data() throws IOException {
		 double[][] arr = new double[MAXINPUTS][MAX_VARIABLES];
		  File file = new File("F:\\guna\\spark\\f3.txt");  //存放数组数据的文件 
		  BufferedReader in = new BufferedReader(new FileReader(file));  //  
		  String line;  //一行数据   
		  line = in.readLine();
		  String[] temp1 = line.split("\t");
		  row_num = Integer.parseInt(temp1[0]);
		  col_num = Integer.parseInt(temp1[1]);
		  input_num = training_cases = row_num;
		  int i,j;
		  i=0;
		  while((line = in.readLine()) != null){  
		   String[] temp2 = line.split("\t");   
		   for(j=0;j<temp2.length;j++){  
		    arr[i][j] = Double.parseDouble(temp2[j]);  
		   }  
		   i++;
		  }  
		  for(i=0;i<row_num;i++) {
			 for(j=0;j<col_num;j++) {
				 training_inputs[i][j]=arr[i][j];
			 }
			 training_outputs[i]=arr[i][j];
		  }
		  in.close(); 
	}

	static CHROMOSOME objective(CHROMOSOME p)
	{
		p.f = 1e10;
		object objecttest = new object();
		long startTime1 = System.currentTimeMillis();
		objecttest.decode_gene(p);
		long endTime1 = System.currentTimeMillis();
		ttime1 += endTime1-startTime1;
		long startTime2 = System.currentTimeMillis();
		objecttest.compute_rule(objecttest.link_root);
		long endTime2 = System.currentTimeMillis();
		ttime2 += endTime2-startTime2;
		double v = 0;
		long startTime3 = System.currentTimeMillis();
		for(int j = 0; j < input_num; j++){
			v += (training_outputs[j] - objecttest.current_value[j])*(training_outputs[j] - objecttest.current_value[j]);	
		}
		long endTime3 = System.currentTimeMillis();
		ttime3 += endTime3-startTime3;
		v = Math.sqrt(v/input_num);
		if(v < 1e-4) v = 0;
			p.f = v;
		if(v < fbest){
			fbest = v;
		}
		evals ++;
		return p;
	}
	
	public static class object{
//		long time11 = System.currentTimeMillis();
		LINK_COMP link_root = new LINK_COMP();
		LINK_COMP[] link_comp=new LINK_COMP[LINK_LENGTH];
		LINK_COMP[] sub_root=new LINK_COMP[GSIZE];
		LINK_COMP sub_comp[][]=new LINK_COMP[GSIZE][GNVARS];
	    double[] current_value=new double[MAXINPUTS];
	    double sub_sibling_value[][]=new double[MAX_SIBLING][MAXINPUTS];
		double[] sub_current_value=new double[MAXINPUTS];
//		long time12 = System.currentTimeMillis();
//		System.out.println("定义变量： "+(time11-time12)+"ms");
		void decode_gene(CHROMOSOME p) {
			long time11 = System.currentTimeMillis();
			int op = -1, i = 0, k = 0, j;
			for(i = 0; i < NVARS; i++){	
				link_comp[i] = new LINK_COMP();
				link_comp[i].value = p.gene[i];
				for(j = 0; j < MAX_SIBLING; j++)
					link_comp[i].siblings[j] = null;
			}
			op = -1; i = 1;
			link_root = link_comp[0];
			if(link_root.value < function_num){
				do{ 
					//find an op type item
					do{op++; if(op >= i)break;}while(link_comp[op].value >= L_terminal);
					if(op >= i) break;
					//set its left and right;
					if(link_comp[op].value < L_terminal){
						if(i >= H+T){break;}
						link_comp[op].siblings[0] = link_comp[i];				
						i++;
						if(link_comp[op].value < 4 || link_comp[op].value >= base_function_num){
							if(i >= H+T){ break;}
							link_comp[op].siblings[1] = link_comp[i];
							i++;
						}
					}
				}while(true);

				if(op < i  && i >= H+T){ 			
					System.out.println("\nERROR RULE111"); 
				}
			}else{
				//printf("terminate");
			}

			//build sub expression trees of the individual
			for(int g = 0; g < GSIZE; g++){
				k = H+T + g *GNVARS;	// the starting position of the ADF.	
				for(i = 0; i < GNVARS; i++){
					sub_comp[g][i] = new LINK_COMP();
					sub_comp[g][i].value =  p.gene[k + i];
					for(j = 0; j < MAX_SIBLING; j++)
						sub_comp[g][i].siblings[j] = null;
				}
				op = -1;
				i = 1;
				sub_root[g] = sub_comp[g][0];
				if(sub_root[g].value < L_terminal){  // note that L_input > L_terminal;
					do{ //find an op type item
						do{op++; if(op >= i)break;}while(sub_comp[g][op].value >= L_terminal);
						if(op >= i) break;
						//set its left and right;
						if(sub_comp[g][op].value < base_function_num){
							if(i >= GH+GT-1){ break;}
							sub_comp[g][op].siblings[0] = sub_comp[g][i];				
							i++;
							if(sub_comp[g][op].value < 4){
								sub_comp[g][op].siblings[1] = sub_comp[g][i];
								i++;
							}
						}
					}while(true);
					if(op < i  && i >= GH+GT - 1){
						System.out.println("SUB ERROR RULE111");
					}
				}else{ 
					//printf("SUB terminate");
				}
			}
			long time12 = System.currentTimeMillis();
//			System.out.println("decode： "+(time12-time11)+"ms");
		}
		void compute_rule(LINK_COMP node) {
			long time11 = System.currentTimeMillis();
			int i;
			if(node.value >= L_terminal){
				for(i = 0; i < input_num; i++)
					current_value[i] = training_inputs[i][node.value - L_terminal];		
			}else{
				double[] t1=new double[MAXINPUTS];
				double[] t2=new double[MAXINPUTS];
				compute_rule(node.siblings[0]);
				for(i = 0; i < input_num; i++) t1[i] = current_value[i];
				if(node.value < 4 || node.value >= base_function_num){
					compute_rule(node.siblings[1]);
					for(i = 0; i < input_num; i++) t2[i] = current_value[i];
				}
				switch(node.value){
				case 0: //+ 			
						for(i = 0; i < input_num; i++) current_value[i] = t1[i] + t2[i]; break;
				case 1: //-
						for(i = 0; i < input_num; i++) current_value[i] = t1[i] - t2[i]; break;
				case 2: //*
						for(i = 0; i < input_num; i++) current_value[i] = t1[i] * t2[i]; break;
				case 3: // /
						for(i = 0; i < input_num; i++){ if(Math.abs(t2[i]) <  1e-20) current_value[i] = 0;else current_value[i] = t1[i] / t2[i];} break;
				case 4: //sin
						for(i = 0; i < input_num; i++){ current_value[i] = Math.sin(t1[i]); } break;
				case 5: //cos
						for(i = 0; i < input_num; i++){ current_value[i] = Math.cos(t1[i]); } break;
				case 6: //exp
						for(i = 0; i < input_num; i++){ if(t1[i] < 20) current_value[i] = Math.exp(t1[i]); else current_value[i] = Math.exp(20.); } break;
				case 7: //log
						for(i = 0; i < input_num; i++){ if(Math.abs(t1[i]) <  1e-20) current_value[i] = 0; else current_value[i] = Math.log(Math.abs(t1[i])); } break;

				default: //GI
						for(i = 0; i < input_num; i++){ sub_sibling_value[0][i] = t1[i]; sub_sibling_value[1][i] = t2[i];}				
						compute_sub_rule(sub_root[node.value - 8]);
						for(i = 0; i < input_num; i++){ current_value[i] = sub_current_value[i];}
						break;
				}
			}
			long time12 = System.currentTimeMillis();
//			System.out.println("comput_rule： "+(time12-time11)+"ms");
		}
		void compute_sub_rule(LINK_COMP node)//子树计算
		{
			long time11 = System.currentTimeMillis();
			int i;
			if(node.value >= L_input){
				// If the node is an input then read data from the input vector, i.e., sub_sibling_value[...];
				for(i = 0; i < input_num; i++) 	sub_current_value[i] = sub_sibling_value[node.value - L_input][i];
				return;
			}else{
				// First compute the left child of the node.
				double[] t1=new double[MAXINPUTS];
				double[] t2=new double[MAXINPUTS];
				compute_sub_rule(node.siblings[0]);

				for(i = 0; i < input_num; i++) t1[i] = sub_current_value[i];
				//then compute the right child of the node if the node contain right child
				if(node.value < 4){  // note that the first 4 functions have 2 children
					compute_sub_rule(node.siblings[1]);
					for(i = 0; i < input_num; i++) t2[i] = sub_current_value[i];
				}
				switch(node.value){
				case 0: //+ 			
						for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] + t2[i]; break;
				case 1: //-
						for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] - t2[i]; break;
				case 2: //*
						for(i = 0; i < input_num; i++) sub_current_value[i] = t1[i] * t2[i]; break;
				case 3: // /
						for(i = 0; i < input_num; i++){ if(Math.abs(t2[i]) <  1e-20) sub_current_value[i] = 0;else sub_current_value[i] = t1[i] / t2[i];} break;
				case 4: //sin
						for(i = 0; i < input_num; i++){ sub_current_value[i] = Math.sin(t1[i]); } break;
				case 5: //cos
						for(i = 0; i < input_num; i++){ sub_current_value[i] = Math.cos(t1[i]); } break;
				case 6: //exp
						for(i = 0; i < input_num; i++){ if(t1[i] < 20) sub_current_value[i] = Math.exp(t1[i]); else sub_current_value[i] = Math.exp(20.); } break;
				case 7: //log
						for(i = 0; i < input_num; i++){ if(Math.abs(t1[i]) <  1e-20) sub_current_value[i] = 0; else sub_current_value[i] = Math.log(Math.abs(t1[i])); } break;
				default: System.out.println("unknow function\n");
				}
			}
			long time12 = System.currentTimeMillis();
//			System.out.println("comput_sub_rule： "+(time12-time11)+"ms");
		}
	}
	
	static int rand_set_value(int I)
	{	
		int x=0;
		switch(gene_type_flag[I]){
		case 0: 
			if(randval(0, 1) < 1./3) x = rand.nextInt(65535)%(base_function_num);		// note that function_num = base_function_num + GSIZE;
			else if(randval(0,1) < 0.5) x = base_function_num + rand.nextInt(65535)%(GSIZE);
			else x = L_terminal + rand.nextInt(65535) % (terminal_num);
			break;
		case 1: x = L_terminal +rand.nextInt(65535) % (terminal_num);
			break;
		case 2: if(rand.nextInt(65535)%2==0)	x = rand.nextInt(65535)%(base_function_num);
			else x = L_input + rand.nextInt(65535)%(2); 
			break;
		case 3:  x = L_input + rand.nextInt(65535)%(2);break;
		default: System.out.println("fds");
		}
		return x;
	}
	
	static double FQ;
	static int MAXIMUM_ELEMENTS=100;
	static double[] function_freq=new double[MAXIMUM_ELEMENTS];						//in the main parts of population, the frequency of each function symbol
	static double[] terminal_freq=new double[MAXIMUM_ELEMENTS];						//in the main parts of population, the frequency of each terminal symbol
	static double[] terminal_probability=new double[MAXIMUM_ELEMENTS];				//store the selection probability of each terminal
	static double[] function_probability=new double[MAXIMUM_ELEMENTS];
	
	static void update_probability()
	{
		double sum = 0;
		int i, j, k;
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

		boolean print_flag = false;
		
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
			sum += function_freq[i];
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
	static int choose_a_terminal()
	{
		int i, j;
		double p = randval(0,1);
		for(i = 0; i < terminal_num - 1; i++){
			if(p < terminal_probability[i])
				break;
		}
		return L_terminal+i;
	}
	//choose a function according to its frequence.
	static int choose_a_function()
	{	
		int i, j, k;
		double p = randval(0,1);
		for(i = 0; i < function_num - 1; i++){
			if(p < function_probability[i])
				break;
		}
		return i;
	}
	//bially set value of bits. 
	static int biasly_set_value(int I)
	{	
		int x = 0;
		//here we only consder the main parts, while the sub-gene part are also randomly setting, so as to import population diversity.
		switch(gene_type_flag[I]){
		case 0: 
			if(randval(0, 1) < FQ) x = choose_a_function();
			else x = choose_a_terminal();
			break;
		case 1: x = choose_a_terminal(); break;
		case 2: 
			if(rand.nextInt(65535)%2==0) x = rand.nextInt(65535)%(base_function_num);
			else x = L_input + rand.nextInt(65535)%(2); 
			break;
		case 3: x = L_input + rand.nextInt(65535)%(2);break;
		default: System.out.println("fds");
		}
		return x;
	}
	
	static void initialize()
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
			population[i] = new CHROMOSOME();
			newpopulation[i] = new CHROMOSOME();
			for(k = 0; k < NVARS; k++){
				population[i].gene[k]=rand_set_value(k);
			}
			population[i].index = i;
			population[i] = objective(population[i]);	
			if(population[i].f < population[ibest].f) ibest = i;	
		}
		population[POPSIZE] = new CHROMOSOME();
		assign(population[ibest],population[POPSIZE]);
	}
	static CHROMOSOME sparkassign(CHROMOSOME oldpop) {
		int  j, k, r1, r2;
		double CR, F;
		double[] change_vector=new double[NVARS];
		F = randval(0, 1);
		CR = randval(0,1);
		do{ r1 = rand.nextInt(65535)%(POPSIZE);}while(r1 == oldpop.index);
		do{r2 = rand.nextInt(65535)%(POPSIZE);}while(r2 == r1 || r2 == oldpop.index);	
		k = rand.nextInt(65535)%(NVARS);
//		System.out.println(oldpop.index);
//		System.out.println(population[oldpop.index].index);
		for(j = 0; j < NVARS; j++){
			if(randval(0,1) < CR || k == j){			
				double dd1 = 0;
				if(((int)population[POPSIZE].gene[j]) != ((int) population[oldpop.index].gene[j])) dd1 = 1;
				double dd2 = 0;
				if(((int)population[r1].gene[j]) != ((int) population[r2].gene[j])) dd2 = 1;
				change_vector[j] = F * dd1 + F * dd2 - (F * dd1 * F * dd2);
				if(randval(0,1) < change_vector[j]){
					oldpop.gene[j]=biasly_set_value(j);
				}else{
					oldpop.gene[j] =  population[oldpop.index].gene[j];
				}
			}else{
				change_vector[j] = 0;
				oldpop.gene[j] = population[oldpop.index].gene[j];
			}
		}
		oldpop = objective(oldpop);
		if(oldpop.f < population[oldpop.index].f){
			assign(oldpop, population[oldpop.index]);
			if(population[oldpop.index].f < population[POPSIZE].f){
				assign(population[oldpop.index], population[POPSIZE]);
			}		
		}
//		System.out.println(oldpop.index);
		return population[oldpop.index];
	}

	static void productiontest()
	{
		for(int i = 0; i < POPSIZE; i++){
			newpopulation[i] = new CHROMOSOME();
			assign(population[i], newpopulation[i]);
		}
		update_probability();
		//生成rdd
		JavaRDD<CHROMOSOME> oldpop = sc.parallelize(Arrays.asList(newpopulation));
		//transform
		oldpop = oldpop.map(x->sparkassign(x));
		
		oldpop.collect();
	}
	static void SLGEP()
	{
		initialize();//初始化
		generation = 0;//代数
		while(generation < 2000){		
			productiontest();//运算
//			if(population[POPSIZE].f < 1e-4){	
//				System.out.println("在第"+generation+"代结束");
//				break;
//			}
			if(generation % 100 == 0)
				System.out.printf("%s\t %s\t %s\t %s\n",function, job, generation, fbest);
			generation++;
		}
	}
	public static void main(String[] args) throws IOException {
		read_data();
		System.setProperty("spark.eventLog.enabled", "true");
		SparkConf conf = new SparkConf().setAppName("sparkga").setMaster("local[*]");
        sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
		long startTime = System.currentTimeMillis();
		SLGEP();			
		long endTime = System.currentTimeMillis();
		System.out.println("程序运行时间： "+(endTime-startTime)+"ms");
		System.out.println("decode： "+ttime1+"ms");
		System.out.println("compute： "+ttime2+"ms");
		System.out.println("compute11： "+ttime3+"ms");
		System.out.println(input_num);
		sc.close();
	}
}

