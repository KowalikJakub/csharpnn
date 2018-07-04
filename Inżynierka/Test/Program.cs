using System;
using MathNet.Numerics.LinearAlgebra;
using MLTools;
namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(0.001f,1f,2,2,2);
            //Console.WriteLine(nn.ToString());
            double[] arr1 = { 0, 0 };
            double[] arr2 = { 0, 1 };
            double[] arr3 = { 1, 1 };
            double[] arr4 = { 1, 0 };
            double[] t1 = { 0 };
            double[] t2 = { 1 };
            double[] t3 = { 0 };
            double[] t4 = { 1 };
            Vector<double> v1 = CreateVector.DenseOfArray(arr1);
            Vector<double> v2 = CreateVector.DenseOfArray(arr2);
            Vector<double> v3 = CreateVector.DenseOfArray(arr3);
            Vector<double> v4 = CreateVector.DenseOfArray(arr4);
            Vector<double> w1 = CreateVector.DenseOfArray(t1);
            Vector<double> w2 = CreateVector.DenseOfArray(t2);
            Vector<double> w3 = CreateVector.DenseOfArray(t3);
            Vector<double> w4 = CreateVector.DenseOfArray(t4);
            Vector<double>[] input = { v1, v2, v3, v4 };
            Vector<double>[] result = { w1, w2, w3, w4 };

            //Console.WriteLine(v1);
            //Console.WriteLine(nn.AddBias(v1)); 
            //Console.WriteLine(nn.FeedForward(v2));

           nn.Train(input, result);

            //Console.WriteLine();
            Console.Write("Input vector\n" + v1.ToString() + "Prediction\n" + nn.FeedForward(v1) + "\nCorrect result\n" + w1.ToString());
            Console.Write("Input vector\n" + v2.ToString() + "Prediction\n" + nn.FeedForward(v2) + "\nCorrect result\n" + w2.ToString());
            Console.Write("Input vector\n" + v3.ToString() + "Prediction\n" + nn.FeedForward(v3) + "\nCorrect result\n" + w3.ToString());
            Console.Write("Input vector\n" + v4.ToString() + "Prediction\n" + nn.FeedForward(v4) + "\nCorrect result\n" + w4.ToString());
        }
    }
}
