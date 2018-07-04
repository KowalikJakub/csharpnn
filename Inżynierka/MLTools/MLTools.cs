using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters;
using System.Runtime.Serialization.Formatters.Binary;

/*
 * TODO:
 * 1. Bias do każdej warstwy
 * 2. Funckję aktywacji dla każdej warstwy (tablica delegatów przekazywana do konstruktora sieci)
 * 3. Algorytm wstecznej propagacji błędu TICK!
 */

namespace MLTools
{
    //public delegate Vector<double> Function(Vector<double> vector);
    [Serializable]
    public class NeuralNetwork : ISerializable
    {
        public class FFResult
        {
            private Vector<double> weightedSum;
            private Vector<double> partialOutput;
            public FFResult(Vector<double> weightedSum, Vector<double> partialOutput)
            {
                this.weightedSum = weightedSum;
                this.partialOutput = partialOutput;
            }
            public Vector<double> WeightedSum { get => weightedSum; set => weightedSum = value; }
            public Vector<double> PartialOutput { get => partialOutput; set => partialOutput = value; }
        }
        private double learningRate;
        private double decayRate;
        private Matrix<double>[] Weights;
        private int inputCount;
        private bool isClassifier;

        public int InputCount { get => inputCount; }
        public bool IsClassifier { get => isClassifier; }
        public double LearningRate { get => learningRate; set => learningRate = value; }
        public double DecayRate { get => decayRate; set => decayRate = value; }

        #region Constructors
        public NeuralNetwork(double learningRate,double decayRate, int inputCount, params int[] LayerCount)
        {
            Weights = new Matrix<double>[LayerCount.Length + 1];
            this.inputCount = inputCount;
            this.learningRate = learningRate;
            this.decayRate = decayRate;

            int rows;
            int columns = inputCount;
            for (int i = 0; i < LayerCount.Length; i++)
            {
                rows = LayerCount[i];
                Weights[i] = CreateMatrix.Random<double>(columns + 1, rows + 1, new MathNet.Numerics.Distributions.ContinuousUniform(0.1f, 0.9f));
                columns = rows;
            }
            Weights[LayerCount.Length] = CreateMatrix.Random<double>(Weights[LayerCount.Length - 1].ColumnCount, 1 , new MathNet.Numerics.Distributions.ContinuousUniform(0.1f, 0.9f));
        }
        //Constructor to deserialize the model
        private NeuralNetwork(SerializationInfo info, StreamingContext context)
        {
            info.GetValue("Weights", typeof(Matrix<double>[]));
            info.GetValue("Learning_Rate", typeof(double));
            info.GetValue("Decay_Rate", typeof(double));
            info.GetValue("Input_Count", typeof(int));
            info.GetValue("Output_Count", typeof(int));
            info.GetValue("Is_Classifier", typeof(bool));
        }
        #endregion

        public Vector<double> FeedForward(Vector<double> Input, double beta = 1f)
        {
            Input = AddBias(Input);
            Vector<double> lhs = Input;
            Matrix<double> rhs;
            for (int i = 0; i < Weights.Length; i++)
            {
                rhs = Weights[i];
                lhs = Functions.Logistic((lhs * rhs), beta);
            }
            return lhs;
        }
        public Vector<double> FeedForward(Vector<double> Input, out List<FFResult> calcInfo, double beta = 1f)
        {
            Input = AddBias(Input);
            calcInfo = new List<FFResult>();
            Vector<double> weightedSum;
            Vector<double> partialOutput;

            Vector<double> lhs = Input;
            Matrix<double> rhs;
            for (int i = 0; i < Weights.Length; i++)
            {
                rhs = Weights[i];
                weightedSum = lhs * rhs;
                partialOutput = Functions.Logistic(weightedSum, beta);
                calcInfo.Add(new FFResult(weightedSum, partialOutput));

                lhs = partialOutput;
            }
            return lhs;
        }
        public void Train(Vector<double>[] trainVectors, Vector<double>[] targets, double beta = 1f)
        {
            if (trainVectors.Length != targets.Length)
                throw new ArgumentException("Rozmiar tablicy wektorów wejściowych i rozmiar tablicy wektorów im odpowiadających nie jest taki sam!");

            List<Vector<double>> deltas = new List<Vector<double>>();
            Random rnd = new Random();
            int randomIndex;
            int EpochCount = 0;
            double error = ComputeError(trainVectors, targets);
            double lastError;

            Stopwatch watch = new Stopwatch();
            watch.Start();
            do
            {
                //Losowanie elementu ze zbioru uczącego
                randomIndex = rnd.Next(0, trainVectors.Length);
                var currentVector = trainVectors[randomIndex];
                var currentTarget = targets[randomIndex];

                //Przebieg przez sieć dla wylosowanego elementu
                var output = FeedForward(currentVector, out List<FFResult> calcInfo, beta);

                //Policzenie delty dla warstwy wyjściowej
                var outDelta = (currentTarget - output).PointwiseMultiply(Derivatives.Logistic(calcInfo[calcInfo.Count - 1].WeightedSum));
                var currentDelta = outDelta;
                deltas.Add(currentDelta);

                //Przebieg wstecz, policzenie delty dla każdej jednostki
                for (int i = calcInfo.Count - 1; i > 0; i--)
                {
                    //currentDelta = (Weights[i] * currentDelta).PointwiseMultiply(Derivatives.Logistic(calcInfo[i - 1].WeightedSum));
                    currentDelta = (Weights[i] * currentDelta).PointwiseMultiply(Derivatives.Logistic(calcInfo[i - 1].WeightedSum));
                    deltas.Add(CreateVector.DenseOfVector(currentDelta));
                }

                //Aktualizacja wag wszystkich macierzy
                for (int i = 0; i < Weights.Length; i++)
                {
                    for (int j = 0; j < Weights[i].RowCount; j++)
                    {
                        for (int k = 0; k < Weights[i].ColumnCount; k++)
                        {
                            Weights[i][j, k] = (1 - learningRate * decayRate) * Weights[i][j, k] - 2 * learningRate * deltas[deltas.Count - 1 - i][k];
                        }
                    }
                }

                //Aktualizacja błędu
                lastError = error;
                error = ComputeError(trainVectors, targets);
                Console.Write("Current Epoch " + EpochCount + '\r');
                EpochCount++;
            } while (error < lastError);

            watch.Stop();
            Console.WriteLine("Training took " + EpochCount + " epochs and " + watch.ElapsedMilliseconds / 1000f + " seconds");
        }
        private double ComputeError(Vector<double>[] trainVectors, Vector<double>[] targets)
        {
            Vector<double> current;
            double error = 0;
            int setSize = trainVectors.Length;
            int vectorSize = targets[0].Count;
            //Compute the error of the network
            for (int i = 0; i < setSize; i++)
            {
                current = FeedForward(trainVectors[i]);
                for (int j = 0; j < vectorSize; j++)
                {
                    error += Math.Pow(current[j] - targets[i][j], 2);
                }
            }
            return error * 0.5f;
        }
        private Vector<double> AddBias(Vector<double> vector)
        {
            Vector<double> result = CreateVector.Dense<double>(vector.Count + 1);
            for (int i = 0; i > result.Count -1; i++)
            {
                result[i] = vector[i - 1];
            }
            result[result.Count - 1] = 1;  //Bias 
            vector = result;
            return vector;
        }
        public void SaveModel(string path)
        {
            if (this != null)
            {
                FileStream fs = new FileStream(path, FileMode.Create);
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs, this);
            }
            else
            {
                throw new NullReferenceException("Obiekt nie został zainicjalizowany");
            }
        }
        public void ReadModel(string path, out NeuralNetwork network)
        {
            FileStream fs = new FileStream(path, FileMode.Open);
            IFormatter formatter = new BinaryFormatter();
            network = (NeuralNetwork)formatter.Deserialize(fs);
        }
        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Weights", Weights);
            info.AddValue("Learning_Rate", learningRate);
            info.AddValue("Decay_Rate", decayRate);
            info.AddValue("Input_Count", inputCount);
            info.AddValue("Is_Classifier", isClassifier);
        }
        public override string ToString()
        {
            string result = "";
            for (int i = 0; i < Weights.Length; i++)
            {
                result += "Layer: " + i + "\n" + Weights[i].ToString() + "\n";
            }
            return result;
        }
    }
}
