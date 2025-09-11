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

## Now, this is a real gym for your computer

Do not run it with many threads, unless you gor plenty of RAM.
It will eat up your whole CPU.
I will have to refactor it somehow, but for now, be warned. In my case 6 threads was safe to run all the measurments.
After you run the "do_the_thing" function, you will get a benchmark data for roughly 1000 points, which are estimated with that technique.
There are multiple factors that affect the result, like number of neighbors, or maximal temporal shift, all that data is collected along with the raw data, so the result dataframe can be saved as a csv file:
```julia
   using CSV
   CSV.write("yourfilename",df)
```










