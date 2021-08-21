import numpy as np

TRACK_COLORS = [(230, 25, 75),
                (60, 180, 75),
                (255, 225, 25),
                (0, 130, 200),
                (245, 130, 48),
                (145, 30, 180),
                (70, 240, 240),
                (240, 50, 230),
                (210, 245, 60),
                (250, 190, 212),
                (0, 128, 128),
                (220, 190, 255),
                (170, 110, 40),
                (255, 250, 200),
                (128, 0, 0),
                (170, 255, 195),
                (128, 128, 0),
                (255, 215, 180),
                (0, 0, 128),
                (128, 128, 128)]

class Track:
    def __init__(self, pose, timestamp):
        self.pose = pose
        self.timestamp = timestamp
    
"""
Tracker: A stateful tracker for associating detections between frames..
https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/calculators/tracker.ts
Default parameters values come from: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/constants.ts
"""
class Tracker:
    def __init__(self, max_tracks, max_age, min_similarity):
        """
        max_tracks: int, 
                The maximum number of tracks that an internal tracker
                will maintain. Note that this number should be set
                larger than maxPoses. How to set this
                number requires experimentation with a given detector,
                but a good starting place is about 3 * maxPoses.
        max_age: int,   
                The maximum duration of time (in milliseconds) that a
                track can exist without being linked with a new detection
                before it is removed. Set this value large if you would
                like to recover people that are not detected for long
                stretches of time (at the cost of potential false
                re-identifications).
        min_similarity: float  
                New poses will only be linked with tracks if the
                similarity score exceeds this threshold.
        
        """
        self.max_tracks = max_tracks
        self.max_age = max_age
        self.min_similarity = min_similarity
        self.tracks = {} # Dict of tracks, key = track_id, value = instance of class Track
        self.next_id = 1

    def apply(self, poses, timestamp):
        # Filters tracks based on their age.
        self.tracks = {id:track  for (id, track) in self.tracks.items() if timestamp - track.timestamp < self.max_age}
        # Sort poses by their scores from most confident to least confident
        poses = sorted(poses, key=lambda body: body.score, reverse=True)
        # Performs a greedy optimization to link detections with tracks. If incoming
        # detections are not linked with existing tracks, new tracks will be created.
        unmatched_track_indices = list(self.tracks.keys())
        unmatched_detection_indices = []
        for i, pose in enumerate(poses):
            if len(unmatched_track_indices) == 0:
                unmatched_detection_indices.append(i)
                continue
            # Assign the detection to the track which produces the highest pairwise
            # similarity score, assuming the score exceeds the minimum similarity
            # threshold.
            max_track_id = -1
            max_sim = -1
            for track_id in unmatched_track_indices:
                sim = self.similarity(pose, self.tracks[track_id])
                if sim >= self.min_similarity and sim > max_sim:
                    max_track_id = track_id
                    max_sim = sim           
            if max_track_id >= 0:
                pose.track_id = max_track_id
                self.update_track(max_track_id, pose, timestamp)
                unmatched_track_indices.remove(max_track_id)
            else:
                unmatched_detection_indices.append(i)
        
        # Spawn new tracks for all unmatched detections.
        for i in unmatched_detection_indices:
            track_id = self.create_track(poses[i], timestamp)
            poses[i].track_id = track_id

        # If there are too many tracks, we keep only the self.max_tracks freshest tracks
        if len(self.tracks) > self.max_tracks:
            sorted_dict = sorted(self.tracks.items(), key=lambda key_value: key_value[1].timestamp, reverse=True)[:self.max_tracks]
            self.tracks = {k:v for k,v in sorted_dict}

        return poses

    def create_track(self, pose, timestamp):
        track_id = self.next_id
        self.tracks[track_id] = Track(pose, timestamp)
        self.next_id += 1
        return track_id

    def update_track(self, track_id, pose, timestamp):
        self.tracks[track_id].pose = pose
        self.tracks[track_id].timestamp = timestamp


        

"""
TrackerIoU, which tracks objects based on bounding box similarity,
currently defined as intersection-over-union (IoU)
https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/calculators/bounding_box_tracker.ts
"""
class TrackerIoU(Tracker):
    def __init__(self, 
                max_tracks = 18, 
                max_age = 1, 
                min_similarity = 0.15
                ):
        """
        max_tracks, max_age, min_similarity: see Tracker docstring
        """
        super().__init__(max_tracks, max_age, min_similarity)

    def similarity(self, pose, track):
        """
        Computes the intersection-over-union (IoU) between a body bounding box and a track.
        Returns The IoU  between the bounding box and the track. This number is
        between 0 and 1, and larger values indicate more box similarity.
        """
        xmin = max(pose.xmin, track.pose.xmin)
        ymin = max(pose.ymin, track.pose.ymin)
        xmax = min(pose.xmax, track.pose.xmax)
        ymax = min(pose.ymax, track.pose.ymax)
        if xmin >= xmax or ymin >= ymax:
            return 0.
        intersection = (xmax - xmin) * (ymax - ymin)
        area_pose = (pose.xmax - pose.xmin) * (pose.ymax - pose.ymin)
        area_track = (track.pose.xmax - track.pose.xmin) * (track.pose.ymax - track.pose.ymin)
        return intersection / (area_pose + area_track - intersection)

"""
TrackerOKS, which tracks poses based on Object Keypoint Similarity. 
This tracker assumes that keypoints are provided in normalized image coordinates.
https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/calculators/keypoint_tracker.ts
"""        
class TrackerOKS(Tracker):
    def __init__(self, 
                max_tracks = 18, 
                max_age = 1, 
                min_similarity = 0.2,
                keypoint_thresh = 0.3,
                keypoint_falloff = np.array([
                                0.026, 0.025, 0.025, 0.035, 0.035, 
                                0.079, 0.079, 0.072, 0.072, 0.062,
                                0.062, 0.107, 0.107, 0.087, 0.087, 
                                0.089, 0.089
                                ]),
                min_keypoints = 4
                ):
        """
        max_tracks, max_age, min_similarity: see Tracker docstring
        keypoint_thresh: float,
                The minimum keypoint confidence threshold. A keypoint is only
                compared in the similarity calculation if both the new detected 
                keypoint and the corresponding track keypoint have confidences 
                above this threshold.
        keypoint_falloff: list of floats,
                Per-keypoint falloff in similarity calculation.
        min_keypoints: int,
                The minimum number of keypoints that are
                necessary for computing similarity. If the number
                of confident keypoints (between a pose and
                track) are under this value, an similarity of 0.0
                will be given.
        """
        super().__init__(max_tracks, max_age, min_similarity)
        self.keypoint_thresh = keypoint_thresh
        self.keypoint_falloff = keypoint_falloff
        self.min_keypoints = min_keypoints

    def similarity(self, pose, track):
        """
        Computes the Object Keypoint Similarity (OKS) between a pose and track.
        This is similar in spirit to the calculation used by COCO keypoint eval:
        https://cocodataset.org/#keypoints-eval
        In this case, OKS is calculated as:
        (1/sum_i d(c_i, c_ti)) * sum_i exp(-d_i^2/(2*a_ti*x_i^2))*d(c_i, c_ti)
        where
            d(x, y) is an indicator function which only produces 1 if x and y
            exceed a given threshold (i.e. keypointThreshold), otherwise 0.
            c_i is the confidence of keypoint i from the new pose
            c_ti is the confidence of keypoint i from the track
            d_i is the Euclidean distance between the pose and track keypoint
            a_ti is the area of the track object (the box covering the keypoints)
            x_i is a constant that controls falloff in a Gaussian distribution,
            computed as 2*keypointFalloff[i].
        Returns The OKS score between the pose and the track. This number is
        between 0 and 1, and larger values indicate more keypoint similarity.
        """
        box_area = self.area(track.pose) 
        if box_area == 0: return 0

        num_valid_kps = 0
        valid_kps_filter = np.logical_and(pose.keypoints_score > self.keypoint_thresh, track.pose.keypoints_score > self.keypoint_thresh)
        pose_kps = pose.keypoints_norm[valid_kps_filter]
        num_valid_kps = len(pose_kps)
        if num_valid_kps < self.min_keypoints:
            return 0
        else:
            track_kps = track.pose.keypoints_norm[valid_kps_filter]
            x = 2 * self.keypoint_falloff[valid_kps_filter][:, None]
            d_squared = np.power(pose_kps-track_kps, 2)
            oks_total = np.sum(np.exp(-d_squared / (2 * box_area * x * x)))
            return oks_total / num_valid_kps

    def area(self, pose):
        """
        Computes the area of a bounding box that tightly covers keypoints.
        """
        kps = pose.keypoints_norm[pose.keypoints_score > self.keypoint_thresh]
        if len(kps) == 0: return 0
        xmin, ymin = np.min(kps, axis=0)
        xmax, ymax = np.max(kps, axis=0)
        return (xmax - xmin) * (ymax - ymin)

    