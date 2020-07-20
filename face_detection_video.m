clc; clear all

the_video = VideoReader('female.mp4');

video_frame= readFrame(the_video);

face_detector = vision.CascadeObjectDetector();

location = step(face_detector , video_frame);

detected_frame =insertShape(video_frame , 'rectangle' , location);

rectangle_to_Points = bbox2points(location(1,:));

feature_points = detectMinEigenFeatures(rgb2gray(detected_frame),'ROI',location);

point_tracker = vision.PointTracker('MaxBidirectionalError',2);

feature_points = feature_points.Location;

initialize(point_tracker,feature_points,detected_frame);

left =100 ;
bottom =100;
width = size(detected_frame,2);
height = size(detected_frame,1);
video_player = vision.VideoPlayer('Position', [left bottom width height]);

previous_points = feature_points;

while hasFrame(the_video)
    
    video_frame = readFrame(the_video);
    [feature_points , isFound]= step(point_tracker,video_frame);
    new_points = feature_points(isFound,:);
    old_points = previous_points(isFound,:);
    
    if size(new_points , 1)>= 2
    [transformed_rectangle , old_points , new_points] =...
        estimateGeometricTransform(old_points,...
        new_points , 'similarity','MaxDistance',4);
    
    rectangle_to_Points = transformPointsForward(transformed_rectangle, rectangle_to_Points);
    reshaped_rectangle = reshape(rectangle_to_Points' , 1 , []);
    
   detected_frame = insertShape(video_frame,'Polygon',reshaped_rectangle, 'linewidth',2 );
   detected_frame = insertMarker(detected_frame, new_points, '+','color','White');
   
   reshaped_rectangle
   
   previous_points = new_points;
   setPoints(point_tracker, previous_points);
    end
    step(video_player , detected_frame);
end

release(video_player);
    






