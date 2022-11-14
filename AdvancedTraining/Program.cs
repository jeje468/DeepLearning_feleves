using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AdvancedTraining
{
    class AdvancedTraining
    {
        const int inputSize = 4;
        const int hiddenNeuronCount = 500;
        const int outputSize = 1;

        readonly Variable x;
        readonly Function y;

        public AdvancedTraining()
        {
            x = Variable.InputVariable(new int[] { inputSize }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, inputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter w2 = new Parameter(new int[] {outputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            y = CNTKLib.Sigmoid(CNTKLib.Times(w2, CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w1, x), b))));
        }

        public void Train(string[] trainData)
        {
            int n = trainData.Length;

            Variable yt = Variable.InputVariable(new int[] {1, outputSize}, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.0000000001, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int i = 0; i < 100; i++)
            {
                double sumLoss = 0;
                double sumEval = 0;
                foreach (var line in trainData)
                {
                    float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        {x, LoadInput(values[3], values[4], values[5], values[6]) },
                        {yt, Value.CreateBatch(yt.Shape, new float[]{values[0]}, DeviceDescriptor.CPUDevice) }
                    };
                    var outputDataMap = new Dictionary<Variable, Value>() { { loss, null} };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", i, sumLoss / n, sumEval / n));
            }
        }

        public float Prediction(float points, float rebounds, float assists, float fg)
        {
            var inputDataMap = new Dictionary<Variable, Value>() { { x, LoadInput(points, rebounds, assists, fg) } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            return outputDataMap[y].GetDenseData<float>(y)[0][0];
        }

        Value LoadInput(float points, float rebounds, float assists, float fg)
        {
            float[] x_store = new float[inputSize];
            x_store[0] = points;
            x_store[1] = rebounds;
            x_store[2] = assists;
            x_store[3] = fg;
            return Value.CreateBatch(x.Shape, x_store, DeviceDescriptor.CPUDevice);
        }
    }

    public class Program
    {
        static string[] trainData = File.ReadAllLines("draft.txt");
        AdvancedTraining app = new AdvancedTraining();

        void Run()
        {
            app.Train(trainData);
            FileTest();
            //ConsoleTest();
        }

        void FileTest()
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;
            foreach (string line in trainData)
            {
                float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                int good = (int)values[0];
                int pred = (int)Math.Round(app.Prediction(values[3], values[4], values[5], values[6]));

                if (pred == good)
                    if (pred == 1)
                        TP++;
                    else
                        TN++;
                else
                    if (pred == 1)
                    FP++;
                else
                    FN++;
            }
            float accuracy = (float)(TP + TN) / (TP + FP + TN + FN);
            float precision = (float)TP / (TP + FP);
            float sensitivity = (float)TP / (TP + FN);
            float F1 = 2 * (precision * sensitivity) / (precision + sensitivity);
            Console.WriteLine(String.Format("True positive:\t{0}\nTrue negative:\t{1}\nFalse positive:\t{2}\nFalse negative:\t{3}", TP, TN, FP, FN));
            Console.WriteLine(String.Format("Accuracy:\t{0}\nPrecision:\t{1}\nSensitivity:\t{2}\nF1 score:\t{3}", accuracy, precision, sensitivity, F1));
        }

        void ConsoleTest()
        {
            while (true)
            {
                Console.Write("Age:");
                float age = float.Parse(Console.ReadLine());
                Console.Write("Height:");
                float height = float.Parse(Console.ReadLine());
                Console.Write("Weight:");
                float weight = float.Parse(Console.ReadLine());
                Console.Write("Salary:");
                float salary = float.Parse(Console.ReadLine());

                //Console.WriteLine("Prediction:" + app.Prediction(age, height, weight, salary));
            }
        }

        static void Main(string[] args)
        {
            new Program().Run();
        }
    }
}
