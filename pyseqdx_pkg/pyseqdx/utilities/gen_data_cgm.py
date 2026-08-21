import numpy as np
import pandas as pd
import torch
import importlib.resources
import time
import datetime

# from tqdm import tqdm

def load_cgm():
    # load CGM data from disk
    file_path = \
        importlib.resources.files('pyseqdx.data').joinpath('CGM.csv')
    if not file_path.is_file():
        raise FileNotFoundError(
            "CGM.csv is not distributed with this code. Follow "
            "the CGM data preparation instructions in the code bundle's "
            "README.md to create it from the Jaeb public dataset."
        )
    with file_path.open('rb') as f:
        df = pd.read_csv(f)
    df = df.assign(dummy_datetime = pd.to_datetime(df.dummy_datetime))
    return df

def get_cgm(
    n_samples, event_ratio = 0.3,
    window_hours = 2, how_soon_hours = 1, 
    event_free_window = True, event_later_than = 0.75,
    train = 0.7, validate = 0.1, test = 0.2, 
    dtype = torch.float32
):
    
    """create CGM data
    Args:
        n_samples: number of samples (combining all train/validate/test approx.)
        event_ratio: ratio of positive samples (approx.)
        window_hours: duration of monitoring window, in hour
        how_soon_hours: a sample is considered positive, if contain or followed
                        by a hypo event no later than how_soon_hours after 
                        the monitoring window ends
        event_free_window: whether to include event in the monitoring window
        event_later_than: only for event_free_window = False, event no earlier
                          than window_hours * event_later_than in a window
    Returns: a list of train/validate/test data, each is a list of 3 elements
        that are 
            the tensor of x and y [n_samples*prop, ?], ?=1 for y and 
            determined by window_hours (1 measure every 5 min);
            a DataFrame of the identifying information for each row, including
                idx_sample, id, segment: index of sample (row), patient, segment;
                sample_start/end: dt of start/end time of the window;
                event_soon: essentiall y, whether an hypo event onset soon;
                incoming_event, incoming_onset_time: index and onset time of next event 
                                             (not necessarily event_soon=True);
    
    The segment means a piece of contiguous measurement with gap no greater than 1hour.
    Hypo event defined as glucose < 60 (mg/dL I believe) for at least 20 minutes.
    
    """
    
    df = load_cgm().dropna()
    # if two measurements gap > 1hr, break and treat as separate segment 
    df = label_segment(df, gap_sec = 3600)
    # interpolate within consecutive measurement segment to 5min bins
    df = bin_and_interp_segment(df)
    # label hypoglycemia events
    df = label_hypoevent(df, low_val=60, duration_sec=1200)
    
    # train/test/validation split here by id & segment so one event can only
    # appear in one of the train/test/validation set.
    ## Identify unique segment and calculate the number of rows per group
    df['id_segment'] = df['id'].astype(str) + '_' + df['segment'].astype(str)
    segment_sizes = df.groupby('id_segment').size()
    ## Shuffle for randomness
    shuffled_segments = segment_sizes.sample(frac=1)
    ## Compute cumulative row counts so after split the number of measures per
    ## dataset is around, e.g., 0.7 : 0.1 : 0.2.
    cumulative_rows = shuffled_segments.cumsum()
    total_rows = cumulative_rows.iloc[-1]
    ## proportion for splits
    ttl = train + validate + test    
    train = train / ttl
    validate = validate / ttl
    test = test / ttl
    train_threshold = total_rows * train
    validation_threshold = total_rows * (train + validate)
    ## Assign segment to train, validation, and test sets
    split_labels = pd.cut(
        cumulative_rows,
        bins=[0, train_threshold, validation_threshold, total_rows],
        labels=['train', 'validate', 'test'],
        include_lowest=True
    )
    ## Merge labels back into the DataFrame
    df['split'] = df['id_segment'].map(split_labels)
    ## splitted df
    ls_split_df = [
        df[df.split == which].copy() \
        for which in ['train', 'validate', 'test']
    ]
    
    # sample snippet
    # np.random.seed(42)
    ls_sampled = [
        sample_window(
            use_df, int(n_samples * prop), event_ratio = event_ratio, 
            window_hours=window_hours, how_soon_hours=how_soon_hours,
            event_free_window=event_free_window, event_later_than=event_later_than
        ) \
        for (use_df, prop) in zip(
            ls_split_df, 
            [train, validate, test]
        )
    ]
    def dfwindow2tsr(df_window):
        
        # label bin5min for later pivot
        df_window['idx_bin5min'] = df_window.groupby('idx_sample').cumcount()
        # start and end of sample window
        df_window['sample_start'] = \
            df_window.groupby('idx_sample')['dummy_datetime'].\
            transform('min')
        df_window['sample_end'] = \
            df_window.groupby('idx_sample')['dummy_datetime'].\
            transform('max')
        # pivot
        index_names = [
            'idx_sample', 'id', 'segment',
            'sample_start', 'sample_end',
            'event_soon', 'incoming_event', 'incoming_onset_time'
        ]
        pivot_df = df_window.pivot(
            index = index_names, 
            columns = 'idx_bin5min', values = 'gl'
        ).reset_index()
        pivot_df.columns.name = None  # Remove the name of the columns
        # assert not pivot_df.isna().values.any(), 'NaN in time series.'
        assert len(pivot_df) == df_window.idx_sample.nunique()
        df_meta = pivot_df[index_names]
        
        tsr_x = torch.tensor(
            pivot_df.drop(index_names, axis = 1).values
        )
        tsr_x = tsr_x.unsqueeze(-1).to(dtype=dtype)
        tsr_y = torch.tensor(1.*(pivot_df['event_soon'].values))
        tsr_y = tsr_y.unsqueeze(-1).to(dtype=dtype)
        
        return tsr_x, tsr_y, df_meta
    
    return [dfwindow2tsr(df_window) for df_window in ls_sampled]

def label_segment(df, gap_sec = 3600):
    
    # add column:
    #   time_diff: float, time till previous measurement of this patient id
    #   segment: int, index of a segment (unique across different id)
    # Note: index reset since we sort by id and time.
    
    # Step 1: Calculate time difference within each 'id' and segment series 
    # based on a threshold (e.g., 1 hour gap)
    gap_threshold = gap_sec  # 1 hour in seconds

    # Sort values by id and dummy_datetime to ensure proper time difference 
    # calculation
    df = df.sort_values(by=['id', 'dummy_datetime'])

    # Calculate time difference within each 'id'
    df['time_diff'] = df.\
        groupby('id')['dummy_datetime'].\
        diff().dt.total_seconds()

    # Create a new segment for each gap larger than the threshold
    # df['segment'] = (df['time_diff'] > gap_threshold).groupby(df['id']).cumsum()
    df['segment'] = (
        (df['time_diff'] > gap_threshold) | np.isnan(df['time_diff'])
    ).cumsum() # unique segment
    ttl_segment = df.groupby('id').segment.nunique().sum()
    print(
        f'Total {ttl_segment} consecutive segments within which the '
        f'measurement gaps are smaller than {gap_sec} seconds.'
    )
    return df.reset_index(drop = True)

def bin_and_interp_segment(df):
    binned_gl = df.set_index(['dummy_datetime']).\
        groupby(['id', 'segment'])['gl'].\
        resample('5min').\
        mean()
    interp_gl = binned_gl.interpolate()
    df_interp = interp_gl.reset_index()
    # recompute time_diff for later use
    df_interp['time_diff'] = df_interp.\
        groupby(['id', 'segment'])['dummy_datetime'].\
        diff().dt.total_seconds()
    return df_interp

def label_hypoevent(df, low_val = 60, duration_sec = 1200):
    
    # add new columns:
    #   low_val_event: bool, whether gl at this row is low
    #   event_onset: bool, whether this row marks an event onset
    #   idx_event: int, index of event, one number for 1 event, -1 for non-event
    #   event_duration: float, event length in seconds.
    
    # Label events where gl < 60
    df['low_val'] = df['gl'] < low_val
    # Step 3: Label low_val_event only for periods where low_val persists for 
    # more than duration. Though, the following loop is not very efficient, 
    # as it label longer events multiple times.
    
    window_threshold = duration_sec 

    # Initialize low_val_event column
    df['low_val_event'] = False
    df['event_onset'] = False
    df['idx_event'] = int(-1)
    df['event_duration'] = 0
    idx_event = int(0)
    
    # Group by 'id' and 'segment' to process each time series independently
    for (id_val, segment_val), group in df.groupby(['id', 'segment']):
        
        row_start = 0
        row_end = 0
        while row_start < len(group):
            if not group.low_val[row_start:].any(): # if no low_val afterward
                break
            # find 1st low_val since row_start
            row_start = int(row_start + group.low_val.iloc[row_start:].argmax())
            # find 1st none low_val since row_start and back 1
            if group.low_val[row_start:].all():
                row_end = len(group) - 1
            else:
                row_end = int(row_start + group.low_val.iloc[row_start:].argmin() - 1)
            
            # print(row_start, row_end)
            # compute duration
            duration = (
                group.dummy_datetime.iloc[row_end] - \
                    group.dummy_datetime.iloc[row_start]
            ).total_seconds()
            # if long then label
            if duration >= window_threshold:
                idx_start = group.index[row_start] # index of the original df
                idx_end = group.index[row_end]
                df.loc[idx_start:idx_end, 'low_val_event'] = True
                df.loc[idx_start, 'event_onset'] = True
                df.loc[idx_start:idx_end, 'idx_event'] = idx_event
                df.loc[idx_start:idx_end, 'event_duration'] = duration
                idx_event = int(idx_event + 1)
                
            row_start = row_end + 1
        
        # LEGACY, inefficient
        # # Calculate cumulative duration of low_val events
        # cumulative_duration = 0
        # event_start_idx = None
        # 
        # # Iterate through the group and calculate the duration of 
        # # consecutive low_val events
        # for i, row in group.iterrows():
        #     if row['low_val']:  # If current row is a low_val point
                
        #         if event_start_idx is None:  # Mark the start of a low_val event
        #             event_start_idx = i
        #             # print('could be a low_val event')
                
        #         # Add the time difference (in seconds) from the previous row
        #         time_diff = row['dummy_datetime'] -\
        #             group.loc[event_start_idx, 'dummy_datetime']
        #         time_diff = time_diff.total_seconds()
        #         cumulative_duration = time_diff
        #         # print(f'event added {time_diff} sec')
                
        #         # If cumulative low_val duration exceeds duration, 
        #         # label it as a low_val_event
        #         if cumulative_duration >= window_threshold:
        #             # Label the entire window as event
        #             df.loc[event_start_idx:i, 'low_val_event'] = True
        #             df.loc[event_start_idx, 'event_onset'] = True
        #             df.loc[event_start_idx:i, 'idx_event'] = idx_event
        #             idx_event = int(idx_event + 1)
        #             # print('event labeled')
        #     else:
        #         # Reset event tracking if the low_val ends
        #         cumulative_duration = 0
        #         event_start_idx = None
    
    print(f'Total {df.event_onset.sum()} events labeled.')
    return df

def low_val_event_soon(since, within_hour, df):
    # Check if there is a low_val event within ? hour after the since time point
    until = since + pd.Timedelta(hours=within_hour)
    # print(until)
    res_df = df[
        (df['dummy_datetime'] > since) & (df['dummy_datetime'] <= until)
    ]['low_val_event'].any()
    return res_df

def label_prelude(df, within_hour = 0.5):
    
    # add new columns:
    #   incoming_event: int, idx of next (or current if in one) event within id-segment
    #   incoming_onset_time: dt, the time of incoming event onset within id-segment
    #   prelude: bool, whether this non-event time is inside prelude (defined 
    #            by within_hour) of next event.
    
    # label prelude of event (? hour before event onset) per id-segment
    
    # label next event idx per id-segment
    # as indicator for all snippets 
    # (separate trajectories by id, segment, and next event).
    # incoming_event = -1 means last piece of a segment with no event following.
    df['incoming_event'] = df.replace({'idx_event': {-1 : pd.NA}}).\
        groupby(['id', 'segment'])['idx_event'].\
        bfill()
    df.loc[df['incoming_event'].isna(), 'incoming_event'] = -1
    df.incoming_event = df.incoming_event.astype('int')
    
    # Identify rows with `event_onset` and assign them as the "incoming_onset_time"
    df['incoming_onset_time'] = df.loc[df['event_onset'], 'dummy_datetime']

    # fill within each segment to assign the next event onset time to earlier rows
    df['incoming_onset_time'] = \
        df.groupby(['id', 'segment', 'incoming_event'])\
            ['incoming_onset_time'].bfill().ffill()
    # # keep NA for those inside an event (for clarity), 
    # # but will cause issue for event_free_window = False
    # df.loc[df.low_val_event, 'incoming_onset_time'] = pd.NA
    # # instead, fill it with current onset time
    
    
    # Calculate the "prelude cutoff" time (30 minutes before `incoming_onset_time`)
    df['prelude_cutoff'] = \
        df['incoming_onset_time'] - datetime.timedelta(hours=within_hour)

    # Label rows as "prelude" if they are before the event onset and after the cutoff
    df['prelude'] = \
        (df['dummy_datetime'] < df['incoming_onset_time']) & \
        (df['dummy_datetime'] >= df['prelude_cutoff']) & \
        (~df['low_val_event']) # and certainly not inside existing event
    return df.drop(['prelude_cutoff'], axis=1)

def sample_window(
    df, n_samples, event_ratio, 
    window_hours = 2, how_soon_hours = 1,
    event_free_window = True, event_later_than = 0.75
):
    # create `n_samples` from `df` with length `window_hours` where 
    # positive/negative is defined by whether an event would onset within
    # `how_soon_hours` since the end of the window interval.
    # If `event_free_window`, sampled window contains no event, otherwise
    # use `event_later_than` to control how soon event could onset in sampled
    # window, e.g., put no earlier than the final 1 - 0.75 = 0.25 portion.
    # Note that in such case the class label will be determined by
    # whether an event onset during or shortly after the monitoring window.
    # 
    # add new columns:
    #   event_soon: whether this sample is positive or not
    # added by `label_prelude`:
    #   incoming_event: int, idx of next (or current if in one) event within id-segment
    #   incoming_onset_time: dt, the time of next event onset within id-segment
    #   prelude: bool, whether this non-event time is inside prelude (defined 
    #            by within_hour) of next event.
    
    assert \
        event_free_window | ((event_later_than < 1) & (event_later_than > 0)) ,\
        "event_later_than should be within 0 and 1."
        
    print(f"Requested event rate {event_ratio}.")
    ttl_event = df.event_onset.sum()
    if n_samples * event_ratio >= ttl_event:
        print(
            f'Around {int(n_samples * event_ratio)} positive samples requested'
            f' from total {ttl_event} events, watch out for repeated/overlappy sample.'
        )
    window_duration = pd.Timedelta(hours=window_hours)  # Define the window
    
    # label prelude
    df = label_prelude(df, how_soon_hours)
    
    # label snippet start time, here a snippet is identified within id-segment
    # by a period of normal measures, potentially with an event in the end.
    df['snippet_start'] = df.groupby(['id', 'segment', 'incoming_event'])\
        ['dummy_datetime'].\
        transform('min')
        
    # label onset_time (basically fill in the NaN during event)
    # for non-event-free window only
    df['snippet_onset_time'] = df.groupby(['id', 'segment', 'incoming_event'])\
        ['incoming_onset_time'].\
        transform('min')
        
    # label all possible end time of monitoring window
    df['endtime_ok'] = (
            # not too close to snippet start
            (df.dummy_datetime > df.snippet_start + window_duration)
        )
    if event_free_window:
        # not during an event
        df['endtime_ok'] = df.endtime_ok & (~ df.low_val_event)
        df['positive_outcome'] = df.prelude
    else:
        # can be during an event not too late into it
        df['endtime_ok'] = df.endtime_ok & \
            (
                df.dummy_datetime < df.snippet_onset_time +\
                    pd.Timedelta(hours = window_hours * (1 - event_later_than))
            )
        df['positive_outcome'] = df.prelude | df.low_val_event
        
    # and that start, end, and onset time is during 06:00 - 24:00
    # though nocturnal hypo also seems important, so include anyway.
    # df['endtime_ok'] = (
    #     df.endtime_ok &\
    #     (df['dummy_datetime'].dt.hour >= 6 + window_hours) &\
    #     (df['dummy_datetime'].dt.hour <= 24 - how_soon_hours)
    # )
    
    # Break if there are no valid end times left
    assert sum(df.endtime_ok) > 0
    
    # upweight to create requested event_ratio in final sample
    positive_ratio = (df.positive_outcome & df.endtime_ok).sum() / df.endtime_ok.sum()
    df['sample_weight'] = (1 - event_ratio) / (1 - positive_ratio)
    df.loc[df.positive_outcome, 'sample_weight'] = event_ratio / positive_ratio
    if (df.positive_outcome & df.endtime_ok).sum() < n_samples * event_ratio:
        sample_w_replacement = True
        print(
            f"{n_samples * event_ratio} positive sample requested while only "
            f"{(df.positive_outcome & df.endtime_ok).sum()} possible, "
            f"using sample with replacement."
        )
    else:
        sample_w_replacement = False
    
    # sample window end time
    df_endtime = df.loc[df.endtime_ok].sample(
        n_samples, weights='sample_weight', replace = sample_w_replacement
    )
    # sort for better presentation
    df_endtime = df_endtime.sort_values(by=['id', 'segment'])
        
    sample_count = 0
    valid_samples = []
    
    for _, row in df_endtime.iterrows():
        snippet = df.loc[
            (df.id == row.id) & (df.segment == row.segment) & \
            (df.incoming_event == row.incoming_event) & \
            (df.dummy_datetime <= row.dummy_datetime) & \
            (df.dummy_datetime >= row.dummy_datetime - window_duration)
        ].copy()
        snippet['event_soon'] = snippet.positive_outcome.max()
        snippet['idx_sample'] = sample_count
        sample_count += 1
        valid_samples.append(snippet)
    
    df_res = pd.concat(valid_samples).drop([
        'snippet_start', 'snippet_onset_time', 'endtime_ok', 
        'positive_outcome', 'sample_weight'
    ], axis=1) # ignore index since we use .copy() and may sample w/ replacement?
    
    print(f"Resulting event rate {df_res.event_soon.mean():.3f}.")
    return df_res

# np.random.seed(42)
def sample_window_old(
    df, n_samples, event_ratio, window_hours = 2, how_soon_hours = 1,
    verbose = False
):
    print(f"Requested event rate {event_ratio}.")
    ttl_event = df.event_onset.sum()
    if n_samples * event_ratio >= ttl_event:
        print(
            f'Around {int(n_samples * event_ratio)} positive samples requested'
            f' from total {ttl_event} events, be careful of repeated sample.'
        )
    how_soon = how_soon_hours
    window_duration = pd.Timedelta(hours=window_hours)  # Define the window
    
    # Randomly select an interval end_time from the dataset, ensuring 
    # it's outside a low_val_event (handle later)
    # and that start, end, and onset time is during 06:00 - 24:00
    df_valid_end_times = df[
        # (~df['low_val_event']) &\
        (df['dummy_datetime'].dt.hour >= 6 + window_hours) &\
        (df['dummy_datetime'].dt.hour <= 24 - how_soon)
    ]

    # Break if there are no valid end times left
    assert not df_valid_end_times.empty

    sample_count = 0
    valid_samples = []
    # Random sampling loop until we reach the desired number of valid samples
    while sample_count < n_samples:
        idx_event = -1
        # Randomly sample an end_time
        if np.random.uniform() <= event_ratio: # sample event onset directly
            df_anchor_time =\
                df_valid_end_times[df_valid_end_times['event_onset']].sample(1)
            idx_event = df_anchor_time['idx_event'].values[0]
            anchor_time = df_anchor_time['dummy_datetime'].values[0]
            from_id = df_anchor_time['id'].unique()[0]
            from_seg = df_anchor_time['segment'].unique()[0]
            df_id = df[df['id'] == from_id]
            df_segment = df_id[df_id['segment'] == from_seg]
            # sample an end_time locally
            df_valid_end_times_local = df_segment[
                (df_segment['dummy_datetime'] < anchor_time) &\
                (df_segment['dummy_datetime'] > (
                    anchor_time - pd.Timedelta(hours=how_soon)
                ))
            ]
            if df_valid_end_times_local.empty: # skip if no candidate time
                continue
            df_end_time = df_valid_end_times_local.sample(1)
        else: # sample outside event
            df_end_time = df_valid_end_times[
                ~df_valid_end_times['low_val_event']
            ].sample(1)
            from_id = df_end_time['id'].unique()[0]
            from_seg = df_end_time['segment'].unique()[0]
            df_id = df[df['id'] == from_id]
            df_segment = df_id[df_id['segment'] == from_seg]
            
        end_time = df_end_time['dummy_datetime'].values[0]
        # Compute the corresponding start_time
        start_time = end_time - window_duration
        
        # Find the segment of this end_time to ensure the window 
        # fits within the segment
        segment_start = df_segment['dummy_datetime'].min()
        pid_end = df_id['dummy_datetime'].max()
        # Skip if the start_time is earlier than the segment start or 
        # end_time + soon later than last measure of the patient
        if start_time < segment_start or \
        end_time + pd.Timedelta(hours=how_soon) > pid_end:
            # print('gave up this end_time')
            continue  # Invalid sample, so continue the loop
        
        # Select the data within this 5-hour window
        subsample = df_segment[
            (df_segment['dummy_datetime'] >= start_time) &\
            (df_segment['dummy_datetime'] <= end_time)
        ].copy()
        
        # if any event in the window, skip
        if subsample['low_val_event'].any():
            continue
        
        # Label whether a low_val_event occurs within ? hours 
        # after the end of the window
        subsample['event_soon'] = \
            low_val_event_soon(end_time, how_soon, df_segment)
        subsample['idx_sample'] = sample_count
        subsample['idx_event'] = idx_event
        
        # Store the subsample if valid
        valid_samples.append(subsample)
        
        if verbose and (1 + sample_count) % 1000 == 0:
            print(
                f'- {time.strftime("%H:%M:%S", time.localtime())}',
                f' sampled {1 + sample_count}.'
            )
            
        # Increment the sample counter
        sample_count += 1

    return pd.concat(valid_samples)

