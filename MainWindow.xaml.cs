using DlibDotNet;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Drawing;
using System.IO;
using System.Media;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
namespace FocusApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture capture;
        private CancellationTokenSource cancelTokenSource;
        private FrontalFaceDetector faceDetector;
        private ShapePredictor shapePredictor;
        private SoundPlayer player;

        private float minAudioDuration = 2;
        private float maxdangerTime = 30;
        private bool areUsing = false;

        private DateTime curDangerTime = DateTime.UtcNow;
        private DateTime curAudioDuration = DateTime.UtcNow;

        private bool isPlaying = false;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void ResetDangerTime()
        {
            curDangerTime = DateTime.UtcNow;
        }

        private float GetDangerTimeSpent()
        {
            TimeSpan duration = DateTime.UtcNow - curDangerTime;
            return duration.Seconds;
        }

        private void ResetAudioTime()
        {
            curAudioDuration = DateTime.UtcNow;
        }

        private float GetTimeListenToAudio()
        {
            TimeSpan duration = DateTime.UtcNow - curAudioDuration;
            return duration.Seconds;
        }
        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            capture = new VideoCapture(0);

            if (capture.IsOpened() == false)
            {
                MessageBox.Show("Could not open webcam");
                return;
            }

            cancelTokenSource = new CancellationTokenSource();
            Task.Run(() => CaptureFrames(cancelTokenSource.Token));

        }


        //This is the main loop
        private async Task CaptureFrames(CancellationToken cancelToken)
        {
            string basePath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = System.IO.Path.Combine(basePath, ".\\DetectionTypes\\shape_predictor_68_face_landmarks.dat");

            player = new SoundPlayer(System.IO.Path.Combine(basePath, ".\\AudioFiles\\Alarm.wav"));
            int frameWidth = (int)capture.Get(VideoCaptureProperties.FrameWidth);
            int frameHeight = (int)capture.Get(VideoCaptureProperties.FrameHeight);
            int prevUpdateFrame = 10;
            int curFrame = 0;
            Mat grayFrame = new Mat();
            List<OpenCvSharp.Point> leftEyePoints = new List<OpenCvSharp.Point>();
            List<OpenCvSharp.Point> rightEyePoints = new List<OpenCvSharp.Point>();

            OpenCvSharp.Point prevLeftPupil = new OpenCvSharp.Point(0, 0);
            OpenCvSharp.Point leftPupil = new OpenCvSharp.Point(0, 0);
            OpenCvSharp.Point prevRightPupil = new OpenCvSharp.Point(0, 0);
            OpenCvSharp.Point rightPupil = new OpenCvSharp.Point(0, 0);


            faceDetector = Dlib.GetFrontalFaceDetector();
            shapePredictor = ShapePredictor.Deserialize(modelPath);
            bool updatedPrev = false;
            using (Mat frame = new Mat())
            {



                while (cancelToken.IsCancellationRequested == false)
                {
                    if ((capture.Read(frame) && !frame.Empty()) == false)
                    {
                        continue;
                    }

                    prevLeftPupil = leftPupil;
                    prevRightPupil = rightPupil;
                    leftEyePoints.Clear();
                    rightEyePoints.Clear();
                    //Grayscale the image
                    Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);


                    byte[] byteArray = new byte[frame.Width * frame.Height * frame.ElemSize()];
                    System.Runtime.InteropServices.Marshal.Copy(frame.Data, byteArray, 0, byteArray.Length);
                    var dlibImg = Dlib.LoadImageData<BgrPixel>(byteArray, (uint)frame.Height, (uint)frame.Width, (uint)frame.Step());
                    var faces = faceDetector.Operator(dlibImg);
                    foreach (var faceRect in faces)
                    {
                        var shape = shapePredictor.Detect(dlibImg, faceRect);
                        //Right eye
                        DrawKeyPointOnFrame(frame, shape, 36, 42, Scalar.Blue);
                        rightEyePoints = GetKeyPoints(shape, 36, 42);
                        //Left eye
                        DrawKeyPointOnFrame(frame, shape, 42, 48, Scalar.Green);
                        leftEyePoints = GetKeyPoints(shape, 42, 48);
                    }
                    leftPupil = GetPupilCenter(grayFrame, leftEyePoints);
                    rightPupil = GetPupilCenter(grayFrame, rightEyePoints);

                    Cv2.Circle(frame, leftPupil, 1, Scalar.Red, thickness: -1);
                    Cv2.Circle(frame, rightPupil, 1, Scalar.Red, thickness: -1);
                    float distance = GetDistance(leftEyePoints, rightEyePoints);

                    bool eyeMoving = IsEyeMoving(prevLeftPupil, leftPupil, distance) && IsEyeMoving(prevRightPupil, rightPupil, distance);
                    bool eyesClosed = IsEyeClosed(leftEyePoints, distance) && IsEyeClosed(rightEyePoints, distance);
                    bool canStartAlarm = (eyeMoving == false || eyesClosed) && areUsing;
                    if(canStartAlarm == false)
                    {
                        ResetDangerTime();
                    }

                    if(areUsing == false && isPlaying)
                    {
                        player.Stop();
                        isPlaying = false;
                    }

                    if (isPlaying == false && canStartAlarm && maxdangerTime < GetDangerTimeSpent())
                    {
                        player.Play();
                        isPlaying = true;
                        ResetAudioTime();
                    } else if(isPlaying && canStartAlarm == false && GetTimeListenToAudio() > minAudioDuration)
                    {
                        player.Stop();
                        isPlaying = false;
                        ResetDangerTime();

                    }

                    string movingText = eyeMoving ? "Moving your eyes" : "You're distracted";
                    string closedText = !eyesClosed ? "Your eyes are open" : "You're falling asleep";
                    string usingText = areUsing ? "Activated" : "Inactive";
                    Cv2.PutText(frame, usingText, new OpenCvSharp.Point(0, 100), HersheyFonts.HersheyPlain, 1, Scalar.Black);
                    Cv2.PutText(frame, $"Distance: {distance}", new OpenCvSharp.Point(0, 200), HersheyFonts.HersheyPlain, 1, Scalar.Black);
                    Cv2.PutText(frame, movingText, new OpenCvSharp.Point(0, 300), HersheyFonts.HersheyPlain, 1, Scalar.Black);
                    Cv2.PutText(frame, closedText, new OpenCvSharp.Point(0, 400), HersheyFonts.HersheyPlain, 1, Scalar.Black);

                    //Locked in after here
                    var bitmapSource = BitmapSourceConverter.ToBitmapSource(frame);
                    bitmapSource.Freeze();

                    Dispatcher.Invoke(() => {
                        WebcamViewer.Source = bitmapSource;
                    });
                    await Task.Delay(33).ConfigureAwait(false);
                }
            }
        }

        //From 0 to 67(68 keypoints in total)
        //0 to 16 jaw
        //17 to 21 right eyebrow
        //22 to 26 left eyebrow
        //nose 27 to 30
        //lower nose 31 to 35
        //right eye 36 to 41
        //left eye 42 to 47
        //outer mouth 48 to 59
        //inner mouth 60 to 67
        private void DrawKeyPointOnFrame(Mat frame, FullObjectDetection shape, uint start, uint end, Scalar color)
        {
            if (end > 67 || start < 0 || end < 0 || start > 67)
            {
                return;
            }
            for (uint i = start; i < end; i++)
            {
                DlibDotNet.Point dlipPoint = shape.GetPart(i);
                OpenCvSharp.Point cvPoint = new OpenCvSharp.Point((int)dlipPoint.X, (int)dlipPoint.Y);
                Cv2.Circle(frame, cvPoint, radius: 3, color: color, thickness: -1);
            }
        }


        //For eye points 0 is the left of the eye, 1 to 2 is the top, 3 is the right of the eye, and 4 to 5 are the bottom of the eye
        private List<OpenCvSharp.Point> GetKeyPoints(FullObjectDetection shape, uint start, uint end)
        {
            List<OpenCvSharp.Point> output = new List<OpenCvSharp.Point>();
            if (end > 67 || start < 0 || end < 0 || start > 67)
            {
                return output;
            }
            for (uint i = start; i < end; i++)
            {
                DlibDotNet.Point dlipPoint = shape.GetPart(i);
                OpenCvSharp.Point cvPoint = new OpenCvSharp.Point((int)dlipPoint.X, (int)dlipPoint.Y);
                output.Add(cvPoint);
            }

            return output;
        }

        private OpenCvSharp.Point GetPupilCenter(Mat grayFrame, List<OpenCvSharp.Point> eyePoints)
        {
            OpenCvSharp.Rect eyeRect = Cv2.BoundingRect(eyePoints);

            int adjustedX = Math.Max(0, eyeRect.X);
            int adjustedY = Math.Max(0, eyeRect.Y);

            // Adjust Width and Height so they don't overshoot the frame size
            int adjustedWidth = Math.Min(grayFrame.Width - adjustedX, eyeRect.Width);
            int adjustedHeight = Math.Min(grayFrame.Height - adjustedY, eyeRect.Height);
            // Validate boundaries to prevent crashing near frame edges
            if (adjustedWidth <= 0 || adjustedHeight <= 0)
            {
                return new OpenCvSharp.Point((int)(grayFrame.Width / 2), (int)(grayFrame.Height / 2));
            }

            OpenCvSharp.Rect safeEyeRect = new OpenCvSharp.Rect(adjustedX, adjustedY, adjustedWidth, adjustedHeight);

            using Mat eyeRoiGray = new Mat(grayFrame, safeEyeRect);

            // 6. Pupil Isolation via Thresholding 
            // Invert the eye image so dark spots (pupil) become white/bright
            using Mat eyeThreshold = new Mat();
            Cv2.Threshold(eyeRoiGray, eyeThreshold, 50, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

            // 7. Find Contours within the Eye ROI
            Cv2.FindContours(eyeThreshold, out OpenCvSharp.Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            if (contours.Length > 0)
            {
                // Find the largest dark contour inside the eye bounding box (likely the pupil/iris)
                var pupilContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();

                if (Cv2.ContourArea(pupilContour) > 5) // Ignore tiny noise blobs
                {
                    // Calculate center of mass or minimum enclosing circle
                    Cv2.MinEnclosingCircle(pupilContour, out Point2f center, out float radius);

                    // Convert local ROI coordinates back to global frame coordinates
                    OpenCvSharp.Point globalCenter = new OpenCvSharp.Point((int)center.X + safeEyeRect.X, (int)center.Y + safeEyeRect.Y);

                    return globalCenter;
                }
            }

            return new OpenCvSharp.Point((int)(grayFrame.Width / 2), (int)(grayFrame.Height / 2));

        }

        private bool IsEyeClosed(List<OpenCvSharp.Point> eyePoints, float screenDistance)
        {
            if (eyePoints.Count == 0)
            {
                return true;
            }
            int minDistance = (int)(48 / screenDistance);
            int distance = Math.Abs((eyePoints[1].Y + eyePoints[2].Y) / 2 - (eyePoints[4].Y + eyePoints[5].Y) / 2);

            return distance <= minDistance;
        }

        private bool IsEyeMoving(OpenCvSharp.Point prevPupilPos, OpenCvSharp.Point curPupilPos, float screenDistance)
        {
            int minDistance = (int)(4);
            
            double distance = Math.Sqrt(Math.Pow(curPupilPos.X - prevPupilPos.X, 2) + Math.Pow(curPupilPos.Y - prevPupilPos.Y, 2));
            return distance >= minDistance;

        }

        private float GetDistance(List<OpenCvSharp.Point> leftEyePoints, List<OpenCvSharp.Point> rightEyePoints)
        {
            if (leftEyePoints.Count == 0 || rightEyePoints.Count == 0)
            {
                return 1;
            }
            float pxToMM = .265f;
            float baseSize = 65;
            float averageSize = Math.Abs(((leftEyePoints[3].X - leftEyePoints[0].X) +  (rightEyePoints[3].X - rightEyePoints[0].X))/2);
            float curSize = averageSize * pxToMM;

            return baseSize / curSize;
        }

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {

            if(isPlaying)
            {
                player.Stop();
            }
            cancelTokenSource?.Cancel();
            capture?.Dispose();
        }

        private void ChnageUseState(object sender, RoutedEventArgs e)
        {
            if(areUsing)
            {
                areUsing = false;
            }
            else
            {
                areUsing = true;
            }
        }
    }
}