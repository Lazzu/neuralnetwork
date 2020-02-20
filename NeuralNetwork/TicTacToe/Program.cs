using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetwork;

namespace TicTacToe
{
    internal static class Program
    {
        private const double TurnPoints = 0.01f;
        private const int Networks = 10000;
        private const int MutationDecrementIterations = 2500;
        private const int UpdateInterval = 10;
        private const int GamesPerIteration = 30;
        
        private static readonly int[] _Layers = {36, 18};
        private static string _OutputFile = null;
        private static string _InputFile = null;
        
        static void Main(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-o")
                {
                    _OutputFile = args[i + 1];
                    i++;
                }

                if (args[i] == "-i")
                {
                    _InputFile = args[i + 1];
                    i++;
                }
            }
            
            List<Network> networkList = new List<Network>(Networks);
            Random random = new Random();
            
            Network straightNetworkBase = new Network(random, 10, 9, _Layers, Math.Tanh);
            double mutationChance = 1.0;
            double mutationValue = 1.0;
            
            /*if (!string.IsNullOrEmpty(_InputFile))
            {
                Console.WriteLine($"Reading progress from {_InputFile}");
                byte[] bytes = File.ReadAllBytes(_InputFile);
                Console.WriteLine($"File read. Creating neural network from the data read from file");
                straightNetworkBase.FromBytes(bytes, 0);
                mutationChance = 0.125;
                mutationValue = 0.125;
                Console.WriteLine($"Done reading.");
            }*/
            
            Console.WriteLine($"Creating networks");
            for (int i = 0; i < Networks; i++)
            {
                double mutationVal = mutationValue;
                straightNetworkBase.Mutate(random, (rand, currentValue) =>
                {
                    return currentValue + (rand.NextDouble() - 0.5) * 2 * mutationVal;
                });
                networkList.Add(straightNetworkBase.CloneNetwork());
            }
            
            mutationChance = 0.5;
            mutationValue = 0.25;
            
            Stopwatch sw = new Stopwatch();

            for (int iteration = 0; true; iteration++)
            {
                networkList.ForEach(net => net.Fitness = 0);
                sw.Restart();
                for(int i = 0; i < GamesPerIteration; i++)
                {
                    Parallel.For(0, Networks - 1, networkIndex =>
                    {
                        Network opponent = networkList[networkIndex + 1];
                        PlayGame(networkList[networkIndex], opponent);
                    });
                    networkList.Sort();
                }
                sw.Stop();

                double playTime = sw.Elapsed.TotalMilliseconds;
                
                int fitNetworkCount = networkList.Count(net => net.Fitness > 0.1);
                int cherryPickedNetworks = fitNetworkCount / 50 + 1;

                if (iteration > 0 && iteration % MutationDecrementIterations == 0)
                {
                    mutationChance /= 2;
                    mutationValue /= 2;
                }
                
                sw.Restart();
                Parallel.For(cherryPickedNetworks - 1, Networks, i => {
                    double fitnessMultiplier = 1.0 / networkList[i].Fitness;
                    int fitNetworkIndex = (i - (cherryPickedNetworks - 1)) % cherryPickedNetworks;
                    networkList[i] = networkList[fitNetworkIndex].CloneNetwork();
                    networkList[i].Mutate(random, (rand, currentValue) => {
                        double chance = 1.0 * fitnessMultiplier;
                        double value = 0.25 * fitnessMultiplier;
                        if (rand.NextDouble() > chance)
                            return currentValue;
                        return currentValue + (rand.NextDouble() - 0.5) * 2 * value;
                    });
                });
                sw.Stop();

                double mutationTime = sw.Elapsed.TotalMilliseconds;

                if (iteration % UpdateInterval == 0)
                {
                    Console.WriteLine($"Fit networks on this iteration: {fitNetworkCount}. " +
                                      $"Cherry-picking {cherryPickedNetworks} networks. " +
                                      $"Mutating {Networks - cherryPickedNetworks} networks. " + 
                                      $"Current mutation chance = {mutationChance:F4}. " +
                                      $"Current mutation value = {mutationValue:F4}. " +
                                      $"Time elapsed per iteration, playing: {playTime:F1} ms, mutating: {mutationTime:F1} ms");
                    Console.Write($"Top 10 networks ");
                    for (int i = 0; i < 10; i++)
                    {
                        Console.Write($"{networkList[i].Fitness:F2}, ");
                    }
                    Console.WriteLine();
                    Console.WriteLine($"Iteration: {iteration}, playing a game with two of the best against each other");
                    PlayGameOnScreen(networkList[0], networkList[1]);
                    /*if (iteration > 0 && !string.IsNullOrEmpty(_OutputFile))
                    {
                        byte[] bytes = networkList[0].ToBytes();
                        File.WriteAllBytes(_OutputFile, bytes);
                    }*/
                }
                
                if (Console.KeyAvailable)
                {
                    break;
                }
            }
            
            networkList.Sort();

            int topNumber = 10;
            if (topNumber > Networks)
            {
                topNumber = Networks;
            }
            
            Console.WriteLine($"Top {topNumber} networks");
            for (int i = 0; i < topNumber; i++)
            {
                Console.WriteLine($"{networkList[i].Fitness:F2}");
            }

            PlayGameOnScreen(networkList[0], networkList[1]);
        }

        private static void PlayGameOnScreen(Network currentStraightNetwork, Network opponent)
        {
            double[] inputs = new double[10];
            double[] outputs = new double[9];

            Console.WriteLine(); 
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            //DisplayBoard(inputs, 0);
            
            inputs[9] = 1.0;
            int turnCount = 0;
            while (turnCount < 9)
            {
                currentStraightNetwork.Forward(inputs, outputs);

                // Find the highest output index
                int highestIndex = -1;
                double highestIndexValue = 0.0;
                for (int i = 0; i < 9; i++)
                {
                    // Is the output activated?
                    if (outputs[i] <= 0.5)
                    {
                        continue;
                    }

                    double value = Math.Abs(outputs[i]);
                    if (highestIndexValue < value)
                    {
                        highestIndex = i;
                        highestIndexValue = value;
                    }
                }

                if (highestIndex < 0)
                {
                    Console.WriteLine("Tried to skip a turn");
                    break;
                }
                
                if (Math.Abs(inputs[highestIndex]) > 0.01)
                {
                    Console.WriteLine($"Tried to put a piece on occupied spot");
                    inputs[highestIndex] = 999;
                    DisplayBoard(inputs, turnCount, 1);
                    break;
                }
                
                inputs[highestIndex] = inputs[9];

                DisplayBoard(inputs, turnCount);

                if (CheckForVictory(inputs, inputs[9]))
                {
                    Console.WriteLine($"Victory!");
                    break;
                }

                inputs[9] = -inputs[9];
                turnCount++;

                Network tmpStraightNetwork = currentStraightNetwork;
                currentStraightNetwork = opponent;
                opponent = tmpStraightNetwork;

                if (turnCount == 9)
                {
                    currentStraightNetwork.Fitness += 0.5;
                    opponent.Fitness += 0.5;
                }
            }
        }

        private static void DisplayBoard(IReadOnlyList<double> inputs, int turn, int vOffset = 0)
        {
            const string X = "X";
            const string O = "O";
            const string ERR = "E";
            const string Y = " ";
            string A = inputs[0] > 990 ? ERR : (inputs[0] > 0 ? X : (inputs[0] < 0 ? O : Y));
            string B = inputs[1] > 990 ? ERR : (inputs[1] > 0 ? X : (inputs[1] < 0 ? O : Y));
            string C = inputs[2] > 990 ? ERR : (inputs[2] > 0 ? X : (inputs[2] < 0 ? O : Y));
            string D = inputs[3] > 990 ? ERR : (inputs[3] > 0 ? X : (inputs[3] < 0 ? O : Y));
            string E = inputs[4] > 990 ? ERR : (inputs[4] > 0 ? X : (inputs[4] < 0 ? O : Y));
            string F = inputs[5] > 990 ? ERR : (inputs[5] > 0 ? X : (inputs[5] < 0 ? O : Y));
            string G = inputs[6] > 990 ? ERR : (inputs[6] > 0 ? X : (inputs[6] < 0 ? O : Y));
            string H = inputs[7] > 990 ? ERR : (inputs[7] > 0 ? X : (inputs[7] < 0 ? O : Y));
            string I = inputs[8] > 990 ? ERR : (inputs[8] > 0 ? X : (inputs[8] < 0 ? O : Y));
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 7 - vOffset);
            Console.Write($"Turn {turn + 1}");
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 6 - vOffset);
            Console.Write("-----");
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 5 - vOffset);
            Console.Write($"|{A}{B}{C}|");
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 4 - vOffset);
            Console.Write($"|{D}{E}{F}|");
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 3 - vOffset);
            Console.Write($"|{G}{H}{I}|");
            Console.SetCursorPosition(turn * 7, Console.WindowHeight - 2 - vOffset);
            Console.WriteLine("-----");
        }

        private static void PlayGame(Network currentStraightNetwork, Network opponent)
        {
            double[] inputs = new double[10];
            double[] outputs = new double[9];
            inputs[9] = 1.0;
            int turnCount = 0;
            while (turnCount < 9)
            {
                currentStraightNetwork.Forward(inputs, outputs);

                // Find the highest output index
                int highestIndex = -1;
                double highestIndexValue = 0.0;
                for (int i = 0; i < 9; i++)
                {
                    // Is the output activated?
                    if (outputs[i] <= 0.5)
                    {
                        continue;
                    }

                    double value = Math.Abs(outputs[i]);
                    if (highestIndexValue < value)
                    {
                        highestIndex = i;
                        highestIndexValue = value;
                    }
                }

                if (highestIndex < 0)
                {
                    // Tried to skip a turn
                    break;
                }
                
                if (Math.Abs(inputs[highestIndex]) > 0.01)
                {
                    // Tried to put a piece on occupied spot
                    currentStraightNetwork.Fitness -= TurnPoints;
                    opponent.Fitness += TurnPoints * (9 - turnCount);
                    inputs[highestIndex] = 999;
                    break;
                }
                
                inputs[highestIndex] = inputs[9];

                if (turnCount == 0 && highestIndex == 4)
                {
                    // Started by putting first on the center, give points!
                    currentStraightNetwork.Fitness += TurnPoints;
                }

                if (CheckForVictory(inputs, inputs[9]))
                {
                    currentStraightNetwork.Fitness += 1;
                    //opponent.Fitness -= 1;
                    break;
                }
                
                currentStraightNetwork.Fitness += TurnPoints;
                //opponent.Fitness += TurnPoints;

                inputs[9] = -inputs[9];
                turnCount++;

                Network tmp = currentStraightNetwork;
                currentStraightNetwork = opponent;
                opponent = tmp;
            }
        }

        private static bool CheckForVictory(IReadOnlyList<double> inputs, double current)
        {
            double[] values = new double[9];
            for (int i = 0; i < 9; i++)
            {
                values[i] = Math.Abs(inputs[i] - current);
            }
            
            /*
             * 012
             * 345
             * 678
             */
            
            if (values[0] < 0.01 && values[1] < 0.01 && values[2] < 0.01)
            {
                return true;
            }
            if (values[3] < 0.01 && values[4] < 0.01 && values[5] < 0.01)
            {
                return true;
            }
            if (values[6] < 0.01 && values[7] < 0.01 && values[8] < 0.01)
            {
                return true;
            }
            if (values[0] < 0.01 && values[3] < 0.01 && values[6] < 0.01)
            {
                return true;
            }
            if (values[1] < 0.01 && values[4] < 0.01 && values[7] < 0.01)
            {
                return true;
            }
            if (values[2] < 0.01 && values[5] < 0.01 && values[8] < 0.01)
            {
                return true;
            }
            if (values[0] < 0.01 && values[4] < 0.01 && values[8] < 0.01)
            {
                return true;
            }
            if (values[6] < 0.01 && values[4] < 0.01 && values[2] < 0.01)
            {
                return true;
            }

            return false;
        }
    }
}