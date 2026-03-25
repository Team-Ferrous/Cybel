## HighNoon Campaign

HighNoon Language Framework was built on CPU first principles for supporting AVX2, AVX512, and NEON for ARM. 
The problem we are having I believe is that we use Tensorflow as a dependency when we aren't utilizing GPUs and the mathematics are focused towards GPU/TPU as well as operations. HighNoon does not support GPUs because we are attempting to pave a pathway to a QPU device so it only makes sense to build it in a way that supports CPU simulations and is fast but also when a QPU becomes avialable we are essentially natively ready for it. 

The goal of this campaign is to completely rewrite the entire HighNoon Language framework using the EID (R&D) loop. We don't even want to assume that this is completely the write path. The campaign will do it's research into the repos I provide it, which will be several. Each repo will have it's own loop that logs all of the files and and directories and analyzes it file by file, and using saguaro tooling. What it is doing is building up a case of how things work, features, etc. That can be used for research and development, and for the feature questionnaire though ideally the model should be going in continuous loops building comprehensive questionnaires for the project and it's features. Believe they're setup to be separate right now which is exactly how I want it. 

This campaign, after building up its research from provided repos, repos it downloads from github, and it's research across the web through reddit, forums, arxiv, etc. It should then build a dev repo that is designed to be fast, loose, and sloppy so to speak. It's goal is to build up and test things, still ensuring the codebase isn't a mess and is complete end to end. There's an existing audit script now that we can probable enhance and build other audit scripts from that are pre versions to the entire thing. For each model layer/sublayer or feature that's going to be in we need to independently test it's mathematics and every other detail we can collect telemetry wise to see it's impact for a model, etc. 

We are challenging all assumptions around current Deep Learning so you must pretend that all research is an assumption until you build and run tests for it to work as intended. The campaigns loop job is to test and build until it lands on a final solution that cannot be improved or enhanced any further. Once the base framework is constructed, it needs to build in depth audit scripts that audit every detail and make it work. Any sudo command installs you will have to have me perform for you since you don't have those permissions and we cannot skip over the hardware telemetry. 

We primarily are after Linux support.

Research needs to be as indepth as possible. 
