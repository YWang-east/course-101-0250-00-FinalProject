# Final project for course **Solving PDEs in Parallel on GPUs**

[![CI action](https://github.com/YWang-east/course-101-0250-00-FinalProject/actions/workflows/CI.yml/badge.svg)](https://github.com/YWang-east/course-101-0250-00-FinalProject/actions/workflows/CI.yml)

This final project consists of two parts:
- [3D diffusion](/scripts-part1/)
- [2D thermomechanical coupling](/scripts-part2)

Part 1 contains only [Diffusion_3D.jl](/scripts-part1/Diffusion_3D.jl), which is a multi-XPUs 3D diffusion solver using implicit time integration and dual-time method. Part 2 contains [TM_2D_prototype.jl](/scripts-part2/TM_2D_prototype.jl) and [TM_2D_perf.jl](/scripts-part2/TM_2D_perf.jl), and they both solve the same thermomechanical coupling problem with the latter one a modified version for parallel execution and better performance. 

For results and more details, please head to respective documentation: [**Part 1**](/docs/part1.md), [**Part 2**](/docs/part2.md).


