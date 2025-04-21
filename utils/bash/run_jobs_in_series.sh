#!/bin/bash

### Copyright 2025 RobotsMali AI4D Lab.

### Licensed under the MIT License; you may not use this file except in compliance with the License.  
### You may obtain a copy of the License at:

### https://opensource.org/licenses/MIT

### Unless required by applicable law or agreed to in writing, software  
### distributed under the License is distributed on an "AS IS" BASIS,  
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
### See the License for the specific language governing permissions and  
### limitations under the License.

echo "Starting training jobs..."

echo "Training "

nohup python train_parakeet_110M.py config_one > train_parakeet_110M.log 2>&1 &
PID1=$!
wait $PID1

nohup python train_parakeet_1B.py config_one > train_parakeet_1B.log 2>&1 &
PID2=$!
wait $PID2

nohup python train_parakeet_10B.py config_one > train_parakeet_10B.log 2>&1 &
PID3=$!
wait $PID3

nohup python train_parakeet_100B.py config_one > train_parakeet_100B.log 2>&1 &
PID4=$!
wait $PID4

echo "All training jobs completed!"
