import torch


class LastKFeatures:
  '''
  Stores last k features for each TrafficSignal.
  '''

  def __init__(self, ts_id_list, feature_shape, k):
    '''
    Args:
      ts_id_list (list[str]): List of TrafficSignal ids.
      feature_shape (tuple(int)): shape of feature.
      k (int): k features to track
    '''
    self.k = k
    self.ts_features = {ts_id: [torch.zeros(feature_shape) for _ in range(k)] for ts_id in ts_id_list }

  def update(self, features):
    '''
    Update self.ts_features with most recent features.

    Args:
      features (dict{str: tensor}): dictionary mapping feature for each TrafficSignal id.
    '''

    for ts_id, feature in features.items():
      self.ts_features[ts_id].insert(0, feature)
      self.ts_features[ts_id].pop()

    

if __name__ == "__main__":
  ts_list = ['ts0', 'ts1', 'ts2']
  k = 2
  last_k = LastKFeatures(ts_list, k)
  print("STEP 1")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.zeros(5), 'ts1': torch.zeros(5), 'ts2': torch.zeros(5)}
  last_k.update(features)
  print("STEP 2")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.ones(5), 'ts1': torch.ones(5), 'ts2': torch.ones(5)}
  last_k.update(features)
  print("STEP 3")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.full(size=(5,),fill_value=2), 'ts1': torch.full(size=(5,),fill_value=2), 'ts2': torch.full(size=(5,),fill_value=2)}
  last_k.update(features)
  print("STEP 4")
  print(f"ts_features: {last_k.ts_features}")
