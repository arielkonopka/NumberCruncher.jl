# NumberCruncher

Welcome to another one of my pet projects! 🚀  
This time we will explore a couple of time series and try to **fiddle with the data they might contain**.  
The main purpose of this project is to **create a data estimator**.

---

## 📦 How to Use This Project

1. **Install Julia**  
   Make sure you have [Julia](https://julialang.org/downloads/) installed.

2. **Clone the repository**  
   ```bash
   git clone https://github.com/arielkonopka/NumberCruncher.jl.git
   ```

3. **Run Julia**  
   Start Julia in the project directory and enter package mode:
   ```julia
   ]
   ```

4. **Activate the environment**  
   ```julia
   activate _your_path_
   ```

5. **Instantiate dependencies**  
   ```julia
   instantiate
   ```

6. **Run the project**  
   Back in Julia REPL:
   ```julia
   using NumberCruncher
   r = NumberCruncher.do_the_thing()
   ```

7. **Now you got the data**
*r* variable contains the dataframe, that was created during the "do_the_thing" function run. You can speed upp your calculations by increasing the number of the threads by invoking:

```bash
julia -t threads
```
Where *threads* is the number of concurrent instances, you would like to allow on your system.

##Your PC's Personal Workout: A High-Intensity Benchmark
Be aware that this program is a resource-intensive benchmark. It will consume a significant amount of your CPU and RAM. It is not recommended to run it with a large number of threads unless you have ample memory. While I intend to refactor it for better efficiency, for now, please proceed with caution. I found that running with six threads was a safe setting for all my measurements.

After executing the *do_the_thing* function, the program will generate benchmark data for approximately 1000 points. The results are a reconstruction of the phase space from the input data, achieved by employing the [PECUZAL (Prediction Error of Coupled Units with a Zonal Adaptive Learning) algorithm](https://iopscience.iop.org/article/10.1088/1367-2630/abe336). 📈

Once the phase space is reconstructed, the next step involves estimating the subsequent data point. This is done by first identifying the neighbors of the last known point. A weighted average is then calculated, where the weights are determined by a combination of the neighbor's distance and the distance to the remaining known coordinates of the neighbor's successor.

All collected data, including factors like the number of neighbors and maximal temporal shift, are included in the output DataFrame. You can easily save this data to a CSV file using the following Julia code:

```Julia

using CSV
CSV.write("yourfilename", df)
```

You will have the data, with the estimated values and real values, along with the distances to the real values. There is also added a visualization part, but it is not ready yet.


