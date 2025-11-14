"""
Empathy Analysis V2.0 - Enhanced Multi-Level Framework
User-level analysis with checkpoint support for large datasets

Author: Analysis Team
Date: 2025-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import re
from collections import Counter
import pickle
import json

# Statistical and ML libraries
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import statsmodels.api as sm

# Suppress warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

# Set random seed
np.random.seed(42)

print("="*80)
print("EMPATHY ANALYSIS V2.0 - USER-LEVEL FRAMEWORK")
print("="*80)
print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print("="*80)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Use relative paths - current directory
BASE_DIR = Path('.')
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# File paths
DATA_FILES = {
    'event_bh': BASE_DIR / 'event_bh.csv',
    'conversations': [
        BASE_DIR / '北海智伴对话250701-0711.xlsx',
        BASE_DIR / '北海智伴对话250712-0719.xlsx',
        BASE_DIR / '北海智伴对话250720-0725.xlsx',
        BASE_DIR / '北海智伴对话250726-0731.xlsx'
    ],
}

# Analysis parameters
CONFIG = {
    'travel_id': 40,  # Beihai travel_id
    'focus_multi_turn': True,
    'min_turns': 3,
    'time_window_minutes': 30,
    'success_window_hours': 24,
    'random_state': 42,
    'test_size': 0.2,
}

print(f"\n📂 Directories:")
print(f"   Base: {BASE_DIR}")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Outputs: {OUTPUT_DIR}")


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(data, checkpoint_name, description=""):
    """Save checkpoint to disk"""
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pkl"
    print(f"\n💾 Saving checkpoint: {checkpoint_name}")
    if description:
        print(f"   Description: {description}")
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Also save metadata
    metadata = {
        'checkpoint_name': checkpoint_name,
        'description': description,
        'timestamp': datetime.now().isoformat(),
        'data_type': str(type(data)),
    }
    
    if isinstance(data, pd.DataFrame):
        metadata['shape'] = data.shape
        metadata['columns'] = list(data.columns)
    elif isinstance(data, dict):
        metadata['keys'] = list(data.keys())
    
    metadata_path = CHECKPOINT_DIR / f"{checkpoint_name}_meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved to: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_name):
    """Load checkpoint from disk"""
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pkl"
    
    if not checkpoint_path.exists():
        return None
    
    print(f"\n📦 Loading checkpoint: {checkpoint_name}")
    
    # Load metadata if exists
    metadata_path = CHECKPOINT_DIR / f"{checkpoint_name}_meta.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"   Saved: {metadata.get('timestamp', 'unknown')}")
        if 'description' in metadata and metadata['description']:
            print(f"   Description: {metadata['description']}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   ✅ Loaded from: {checkpoint_path}")
    return data


def checkpoint_exists(checkpoint_name):
    """Check if checkpoint exists"""
    return (CHECKPOINT_DIR / f"{checkpoint_name}.pkl").exists()


# ============================================================================
# EMPATHY KEYWORDS AND CONTEXT DEFINITIONS
# ============================================================================

EMPATHY_KEYWORDS = {
    'cognitive': {
        'rules_policies': [
            'rule', 'policy', 'regulation', 'time window', 'service fee', 
            'change booking', 'refund', 'prohibited', 'not allowed', 'check-in',
            'typhoon', 'weather', 'service suspension', 'announcement', 'notice'
        ],
        'understanding_signals': [
            'i see what you mean', 'you are saying', 'your question is',
            'understand your', 'in other words', 'to clarify', 'let me understand',
            'so you mean', 'if i understand correctly', 'from your perspective'
        ],
        'context_awareness': [
            'given the weather', 'due to typhoon', 'considering the situation',
            'based on current conditions', 'as announced', 'under these circumstances',
            'in this case', 'for your situation'
        ]
    },
    'affective': {
        'emotion_recognition': [
            'understand', 'sorry', 'apologize', 'appreciate your patience',
            'that must be', 'i can see why', 'that sounds', 'i hear you',
            'i know it is', 'must be difficult'
        ],
        'reassurance': [
            'do not worry', 'no need to worry', 'rest assured', 'we will help',
            'it is okay', 'we can resolve', 'let me help', 'i am here',
            'no problem', 'we can work this out', 'everything will be fine'
        ],
        'validation': [
            'understand your concern', 'that is understandable', 'valid point',
            'appreciate your', 'thank you for', 'that makes sense',
            'you have a point', 'i get it', 'totally understand'
        ]
    },
    'concerns': {
        'action_guidance': [
            'entry', 'click', 'path', 'navigate to', 'mini-program', 'phone', 
            'step', 'procedure', 'subscribe', 'follow announcement', 
            'contact online service', 'change booking', 'you can find', 
            'located at', 'go to', 'open', 'access'
        ],
        'proactive_help': [
            'i can help', 'let me', 'i will check', 'i will find out',
            'allow me to', 'i will assist', 'i will look into', 'let me see',
            'i can arrange', 'i will handle'
        ],
        'options_offering': [
            'you can', 'option', 'alternatively', 'or you can',
            'another way', 'choice', 'either', 'you could also',
            'there is also', 'you may'
        ]
    }
}

MICRO_SKILLS = {
    'clarify': [
        'to confirm', 'please provide', 'could you clarify', 'verify', 'confirm',
        'do you mean', 'to check', 'for verification', 'please specify',
        '为了确认', '请提供', '能否说明', '核对', '确认下', '请问你是指',
        '确认一下', '需要核实', '麻烦提供'
    ],
    'structure_steps': [
        'step', 'first', 'second', 'then', 'next', 'finally', 'lastly',
        '1.', '2.', '3.', '- ', 'procedure', 'process', 'follow these',
        '步骤', '首先', '其次', '然后', '接下来', '最后', '第一', '第二',
        '流程', '按照', '依次'
    ],
    'path_entry': [
        'entry', 'path', 'click', 'mini-program', 'my order', 'waiting hall',
        'seat query', 'change booking', 'navigate to', 'go to', 'open', 'access',
        '入口', '路径', '点击', '小程序', '我的订单', '候船厅',
        '座位查询', '改签', '导航', '前往', '打开', '访问', '进入'
    ],
    'verify_confirm': [
        'is this correct', 'does this work', 'let me confirm', 'verify for you',
        'double check', 'make sure', 'to ensure',
        '是否正确', '这样可以吗', '确认一下', '帮你核实',
        '再次确认', '确保', '保证'
    ],
    'risk_disclaim': [
        'may be affected', 'subject to official announcement', 'not guaranteed',
        'uncertainty exists', 'disclaimer', 'conditions apply', 'subject to change',
        'depends on', 'official notice',
        '可能受影响', '以官方公告为准', '不保证', '存在不确定性',
        '免责', '条件限制', '可能变化', '取决于', '官方通知',
        '实际情况', '以实际为准'
    ],
    'emotion_validation': [
        'understand your', 'can understand', 'appreciate that', 'no need to worry',
        'do not worry', 'rest assured', 'we are here to help', 'i see', 'makes sense',
        '理解你的', '能理解', '辛苦你了', '别着急', '别担心', '放心',
        '我们会帮', '明白', '有道理', '可以理解'
    ],
    'offer_alternative': [
        'if not', 'backup plan', 'plan b', 'alternatively', 'you can also',
        'another option', 'change booking', 'customer service', 'or you could',
        '如果不行', '备用方案', 'B计划', '或者你可以', '改签',
        '客服', '人工', '另外', '备选'
    ]
}

CONTEXT_TRIGGERS = {
    'weather_alert': ['typhoon', 'weather', 'storm', 'wind', 'suspension', 'forecast'],
    'service_disruption': ['delay', 'cancelled', 'suspended', 'not available', 'closed'],
    'urgent_situation': ['urgent', 'immediately', 'right now', 'asap', 'soon', 'hurry'],
    'negative_emotion': ['terrible', 'awful', 'worst', 'unacceptable', 'angry', 'frustrated']
}

USER_SIGNALS = {
    'anxiety': [
        'what should i do', 'will it affect', 'too late', 'worried', 'afraid', 'concerned',
        'nervous', 'unsure', '怎么办', '怎么搞', '怎么弄', '会不会影响', '影响不影响', 
        '有影响吗', '来不及', '太晚', '晚了', '担心', '害怕', '怕', '焦虑', '着急', '急',
        '不安', '忐忑', '紧张', '不确定', '不知道', '不清楚', '有点慌', '慌', '怎么整'
    ],
    'frustration': [
        'terrible', 'how come', 'why', 'not working', 'cannot', 'impossible',
        'ridiculous', 'this is bad', '糟糕', '太糟', '怎么搞的', '为什么', '怎么回事', 
        '怎么会', '不行', '不好用', '用不了', '不能', '没法', '不可以', '不可能', 
        '怎么可能', '不会吧', '离谱', '太离谱', '无语', '郁闷', '烦', '烦死了',
        '崩溃', '受不了', '气死'
    ],
    'urgency': [
        'immediately', 'now', 'urgent', 'hurry', 'running out of time', 'asap',
        'quickly', 'fast', '马上', '立刻', '立即', '现在', '赶紧', '快点', '紧急', 
        '很急', '着急', '赶时间', '来不及', '赶不上', '尽快', '最快', '越快越好',
        '催', '赶', '急需'
    ],
    'gratitude': [
        'thank you', 'thanks', 'appreciate', 'helpful', 'great', 'awesome', 'perfect', 
        'excellent', '谢谢', '感谢', '多谢', '谢了', '太谢谢了', '帮大忙了', '帮了我',
        '有帮助', '很有用', '太好了', '完美', '棒', '赞', '厉害', '牛', '强',
        '满意', '不错', '好评'
    ],
    'relief': [
        'okay', 'good', 'i see', 'understood', 'got it', 'clear', 'makes sense',
        '好的', '好', '明白了', '懂了', '知道了', '清楚了', '了解', '原来如此',
        '那就好', '放心了', '没事了', '解决了'
    ]
}

# Success event definitions
SUCCESS_EVENTS = [
    '来游吧-旅游智能伙伴-点击服务名称',
    '来游吧-旅游智能伙伴-调用服务名称',
    '来游吧-通用旅游智能伙伴-调用服务名称',
    '来游吧-旅游智能伙伴-点击查询座位',
    '来游吧-旅游智能伙伴-模拟用户输入'
]

print("\n✅ Configuration and keywords loaded")


# ============================================================================
# STEP 1: LOAD AND MAP USER IDS
# ============================================================================

def load_user_mapping():
    """Load event data and create session_id -> user_id mapping"""
    checkpoint_name = "01_user_mapping"
    
    # Check checkpoint
    if checkpoint_exists(checkpoint_name):
        return load_checkpoint(checkpoint_name)
    
    print("\n" + "="*80)
    print("STEP 1: LOADING USER MAPPING FROM EVENT DATA")
    print("="*80)
    
    print(f"\n📂 Loading: {DATA_FILES['event_bh'].name}")
    
    # Load session_id and user_id columns
    df_events_mapping = pd.read_csv(
        DATA_FILES['event_bh'], 
        usecols=['session_id', 'user_id']
    )
    
    print(f"   Total event records: {len(df_events_mapping):,}")
    
    # Create mapping: session_id -> user_id (take first user_id per session)
    session_to_user = df_events_mapping.groupby('session_id')['user_id'].first().to_dict()
    
    print(f"   ✅ Created mapping for {len(session_to_user):,} sessions")
    print(f"   ✅ Covering {len(set(session_to_user.values())):,} unique users")
    
    # Save checkpoint
    save_checkpoint(session_to_user, checkpoint_name, 
                    "session_id -> user_id mapping from event_bh.csv")
    
    return session_to_user


# ============================================================================
# STEP 2: LOAD CONVERSATION DATA
# ============================================================================

def load_conversations(session_to_user):
    """Load conversation data and extract user_id from session_id string"""
    checkpoint_name = "02_conversations"
    
    # Check checkpoint
    if checkpoint_exists(checkpoint_name):
        return load_checkpoint(checkpoint_name)
    
    print("\n" + "="*80)
    print("STEP 2: LOADING CONVERSATION DATA")
    print("="*80)
    
    conversations_list = []
    
    for conv_file in DATA_FILES['conversations']:
        if conv_file.exists():
            print(f"\n📂 Loading: {conv_file.name}")
            try:
                df_conv = pd.read_excel(conv_file)
                
                # Print columns for debugging (first file only)
                if len(conversations_list) == 0:
                    print(f"   📋 Columns: {', '.join(df_conv.columns[:10])}{'...' if len(df_conv.columns) > 10 else ''}")
                
                # Filter for Beihai data
                if 'travel_id' in df_conv.columns:
                    df_conv = df_conv[df_conv['travel_id'] == CONFIG['travel_id']].copy()
                
                # Clean data
                if 'im_content' in df_conv.columns:
                    df_conv = df_conv[df_conv['im_content'].notna()].copy()
                    df_conv = df_conv[df_conv['im_content'].str.strip().str.len() >= 5].copy()
                
                conversations_list.append(df_conv)
                print(f"   Records: {len(df_conv):,}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"⚠️ File not found: {conv_file.name}")
    
    if not conversations_list:
        raise ValueError("No conversation data loaded!")
    
    # Concatenate all conversations
    df_conversations = pd.concat(conversations_list, ignore_index=True)
    
    # Convert timestamps
    if 'create_time' in df_conversations.columns:
        df_conversations['create_time'] = pd.to_datetime(
            df_conversations['create_time'], errors='coerce'
        )
    
    # Extract user_id from session_id
    print("\n📌 Extracting user_id from session_id...")
    if 'session_id' not in df_conversations.columns:
        raise ValueError("session_id column not found in conversation data!")
    
    # Session_id format is: {user_id}_{timestamp}
    # Extract the user_id part (before the underscore)
    def extract_user_id(session_id_str):
        if pd.isna(session_id_str):
            return None
        try:
            # Split by underscore and take the first part
            user_id_str = str(session_id_str).split('_')[0]
            # Convert to int to match event data format
            return int(user_id_str)
        except:
            return None
    
    df_conversations['user_id'] = df_conversations['session_id'].apply(extract_user_id)
    
    # Get valid user IDs from event data for filtering
    valid_user_ids = set(session_to_user.values())
    
    # Filter to only include users that exist in event data
    total_records = len(df_conversations)
    df_conversations = df_conversations[df_conversations['user_id'].isin(valid_user_ids)].copy()
    filtered_records = len(df_conversations)
    
    # Data quality check
    matching_rate = filtered_records / total_records * 100 if total_records > 0 else 0
    
    print(f"\n✅ Conversation data loaded:")
    print(f"   Total records before filtering: {total_records:,}")
    print(f"   Records with matching user_id: {filtered_records:,}")
    print(f"   Unique dialogues (im_id): {df_conversations['im_id'].nunique():,}")
    print(f"   Unique sessions: {df_conversations['session_id'].nunique():,}")
    print(f"   Unique users: {df_conversations['user_id'].nunique():,}")
    print(f"   Matching rate: {matching_rate:.1f}%")
    
    if matching_rate < 50:
        print(f"   ⚠️ Warning: Low matching rate")
        print(f"   💡 Sample conversation user_ids: {list(df_conversations['user_id'].head(3))}")
        print(f"   💡 Total valid users in events: {len(valid_user_ids):,}")
    
    # Save checkpoint
    save_checkpoint(df_conversations, checkpoint_name,
                   "Full conversation data with user_id extracted from session_id")
    
    return df_conversations


# ============================================================================
# STEP 3: EMPATHY SCORING FUNCTIONS
# ============================================================================

def calculate_empathy_scores(text, speaker_type='bot', context_flags=None):
    """Calculate three-dimensional empathy scores"""
    if not text or not isinstance(text, str):
        return {
            'cognitive_empathy': 0,
            'affective_empathy': 0,
            'empathy_concerns': 0,
            'total_empathy': 0
        }
    
    text_lower = text.lower()
    
    # Cognitive empathy
    cognitive_score = 0
    for category, keywords in EMPATHY_KEYWORDS['cognitive'].items():
        cognitive_score += sum(1 for kw in keywords if kw.lower() in text_lower)
    
    # Affective empathy
    affective_score = 0
    for category, keywords in EMPATHY_KEYWORDS['affective'].items():
        affective_score += sum(1 for kw in keywords if kw.lower() in text_lower)
    
    # Empathy concerns
    concerns_score = 0
    for category, keywords in EMPATHY_KEYWORDS['concerns'].items():
        concerns_score += sum(1 for kw in keywords if kw.lower() in text_lower)
    
    # Context-aware bonuses
    if context_flags:
        if context_flags.get('weather_alert'):
            cognitive_score *= 1.2
        if context_flags.get('urgent_situation'):
            affective_score *= 1.2
    
    # Normalize
    total = cognitive_score + affective_score + concerns_score
    
    return {
        'cognitive_empathy': cognitive_score,
        'affective_empathy': affective_score,
        'empathy_concerns': concerns_score,
        'total_empathy': total
    }


def calculate_micro_skills(text, speaker_type='bot'):
    """Detect micro-skills in text"""
    if not text or not isinstance(text, str):
        return {skill: 0 for skill in MICRO_SKILLS.keys()}
    
    text_lower = text.lower()
    
    scores = {}
    for skill, keywords in MICRO_SKILLS.items():
        scores[skill] = sum(1 for kw in keywords if kw.lower() in text_lower)
    
    return scores


def detect_context_triggers(texts):
    """Detect context triggers across texts"""
    combined_text = ' '.join([str(t).lower() for t in texts if t])
    
    flags = {}
    for context, keywords in CONTEXT_TRIGGERS.items():
        flags[context] = any(kw in combined_text for kw in keywords)
    
    return flags


def detect_user_emotion(text):
    """Detect user emotional signals"""
    if not text or not isinstance(text, str):
        return {emotion: 0 for emotion in USER_SIGNALS.keys()}
    
    text_lower = text.lower()
    
    scores = {}
    for emotion, keywords in USER_SIGNALS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in text_lower)
    
    return scores


print("\n✅ Empathy scoring functions defined")


# ============================================================================
# STEP 4: MERGE CONSECUTIVE TURNS
# ============================================================================

def merge_consecutive_turns(df_conversations):
    """Merge consecutive messages by the same speaker"""
    checkpoint_name = "03_turns_merged"
    
    # Check checkpoint
    if checkpoint_exists(checkpoint_name):
        return load_checkpoint(checkpoint_name)
    
    print("\n" + "="*80)
    print("STEP 3: MERGING CONSECUTIVE TURNS")
    print("="*80)
    
    df = df_conversations.copy()
    
    # Determine speaker type
    if 'im_type' in df.columns:
        df['speaker_type'] = df['im_type'].apply(
            lambda x: 'bot' if '系统' in str(x) else 'user'
        )
    else:
        df['speaker_type'] = 'user'
    
    # Sort by session and time
    df = df.sort_values(['session_id', 'create_time']).reset_index(drop=True)
    
    # Mark speaker changes
    df['speaker_change'] = (
        (df['speaker_type'] != df['speaker_type'].shift(1)) |
        (df['session_id'] != df['session_id'].shift(1))
    )
    df['turn_id'] = df.groupby('session_id')['speaker_change'].cumsum()
    
    # Aggregate by turn
    print("Aggregating messages into turns...")
    df_turns = df.groupby(['session_id', 'turn_id', 'speaker_type'], as_index=False).agg({
        'im_content': lambda x: ' '.join(x.astype(str)),
        'create_time': ['min', 'max'],
        'user_id': 'first',
        'im_id': 'first'
    })
    
    # Flatten column names
    df_turns.columns = [
        'session_id', 'turn_id', 'speaker_type', 'content',
        'turn_start_time', 'turn_end_time', 'user_id', 'im_id'
    ]
    
    print(f"\n✅ Turn merging completed:")
    print(f"   Original messages: {len(df):,}")
    print(f"   Merged turns: {len(df_turns):,}")
    print(f"   Unique users: {df_turns['user_id'].nunique():,}")
    
    # Save checkpoint
    save_checkpoint(df_turns, checkpoint_name,
                   "Conversations merged into turns by speaker")
    
    return df_turns


# ============================================================================
# STEP 5: CALCULATE USER-LEVEL FEATURES
# ============================================================================

def calculate_user_features(user_id, turn_data):
    """
    Calculate comprehensive user-level features across all conversations
    
    A user may have multiple sessions/dialogues. We aggregate across all.
    """
    user_turns = turn_data[turn_data['user_id'] == user_id]
    
    if len(user_turns) == 0:
        return None
    
    # Separate user and bot turns
    user_messages = user_turns[user_turns['speaker_type'] == 'user']
    bot_turns = user_turns[user_turns['speaker_type'] == 'bot']
    
    # Detect context across all user's conversations
    all_texts = user_turns['content'].tolist()
    context_flags = detect_context_triggers(all_texts)
    
    # Calculate empathy scores for bot responses
    bot_empathy_scores = []
    bot_micro_skills = []
    
    for idx, turn in bot_turns.iterrows():
        empathy = calculate_empathy_scores(
            turn['content'], 
            speaker_type='bot',
            context_flags=context_flags
        )
        micro_skills = calculate_micro_skills(turn['content'], speaker_type='bot')
        
        bot_empathy_scores.append(empathy)
        bot_micro_skills.append(micro_skills)
    
    # Aggregate bot empathy
    if bot_empathy_scores:
        avg_cognitive = np.mean([s['cognitive_empathy'] for s in bot_empathy_scores])
        avg_affective = np.mean([s['affective_empathy'] for s in bot_empathy_scores])
        avg_concerns = np.mean([s['empathy_concerns'] for s in bot_empathy_scores])
        avg_total_empathy = np.mean([s['total_empathy'] for s in bot_empathy_scores])
    else:
        avg_cognitive = avg_affective = avg_concerns = avg_total_empathy = 0
    
    # Aggregate micro-skills
    if bot_micro_skills:
        avg_micro_skills = {
            skill: np.mean([ms[skill] for ms in bot_micro_skills])
            for skill in MICRO_SKILLS.keys()
        }
    else:
        avg_micro_skills = {skill: 0 for skill in MICRO_SKILLS.keys()}
    
    # Detect user emotions (first and last turn)
    if len(user_messages) > 0:
        first_user_emotion = detect_user_emotion(user_messages.iloc[0]['content'])
        last_user_emotion = detect_user_emotion(user_messages.iloc[-1]['content'])
        
        # Emotion shift
        initial_negative = first_user_emotion['anxiety'] + first_user_emotion['frustration']
        final_positive = last_user_emotion['relief'] + last_user_emotion['gratitude']
        emotion_shift = final_positive - initial_negative
    else:
        emotion_shift = 0
        first_user_emotion = {k: 0 for k in USER_SIGNALS.keys()}
        last_user_emotion = {k: 0 for k in USER_SIGNALS.keys()}
    
    # Calculate response times
    response_times = []
    for i, (idx, user_msg) in enumerate(user_messages.iterrows()):
        next_bot = bot_turns[bot_turns['turn_start_time'] > user_msg['turn_end_time']]
        if len(next_bot) > 0:
            response_time = (next_bot.iloc[0]['turn_start_time'] - 
                           user_msg['turn_end_time']).total_seconds()
            response_times.append(response_time)
    
    avg_response_time = np.mean(response_times) if response_times else 0
    
    # Conversation characteristics (across all user's sessions)
    num_turns = len(user_turns)
    num_user_turns = len(user_messages)
    num_bot_turns = len(bot_turns)
    num_sessions = user_turns['session_id'].nunique()
    num_dialogues = user_turns['im_id'].nunique() if 'im_id' in user_turns.columns else 0
    
    # Duration (across all conversations)
    start_time = user_turns['turn_start_time'].min()
    end_time = user_turns['turn_end_time'].max()
    total_duration = (end_time - start_time).total_seconds()
    
    # Compile all features
    features = {
        'user_id': user_id,
        'num_sessions': num_sessions,
        'num_dialogues': num_dialogues,
        
        # Empathy dimensions
        'cognitive_empathy': avg_cognitive,
        'affective_empathy': avg_affective,
        'empathy_concerns': avg_concerns,
        'total_empathy': avg_total_empathy,
        
        # Micro-skills
        **{f'skill_{k}': v for k, v in avg_micro_skills.items()},
        
        # Context
        **{f'context_{k}': int(v) for k, v in context_flags.items()},
        
        # User emotions
        'initial_anxiety': first_user_emotion['anxiety'],
        'initial_frustration': first_user_emotion['frustration'],
        'initial_urgency': first_user_emotion['urgency'],
        'final_relief': last_user_emotion['relief'],
        'final_gratitude': last_user_emotion['gratitude'],
        'emotion_shift': emotion_shift,
        
        # Conversation characteristics
        'num_turns': num_turns,
        'num_user_turns': num_user_turns,
        'num_bot_turns': num_bot_turns,
        'total_duration_seconds': total_duration,
        'avg_response_time': avg_response_time,
    }
    
    return features


def analyze_users(df_turns):
    """Analyze all users and calculate features"""
    checkpoint_name = "04_user_features"
    
    # Check checkpoint
    if checkpoint_exists(checkpoint_name):
        return load_checkpoint(checkpoint_name)
    
    print("\n" + "="*80)
    print("STEP 4: CALCULATING USER-LEVEL FEATURES")
    print("="*80)
    
    # Filter conversations with valid user_id
    df_turns_valid = df_turns[df_turns['user_id'].notna()].copy()
    print(f"Turns with valid user_id: {len(df_turns_valid):,} / {len(df_turns):,}")
    
    # Get unique users
    unique_users = df_turns_valid['user_id'].unique()
    total_users = len(unique_users)
    
    print(f"\n Processing {total_users:,} unique users...")
    print("(Each user may have multiple sessions/dialogues)")
    
    # Calculate features for each user
    user_features_list = []
    
    for i, user_id in enumerate(unique_users):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1:,}/{total_users:,} ({(i+1)/total_users*100:.1f}%)")
        
        features = calculate_user_features(user_id, df_turns_valid)
        if features:
            user_features_list.append(features)
    
    df_user_features = pd.DataFrame(user_features_list)
    
    print(f"\n✅ User analysis completed:")
    print(f"   Total users analyzed: {len(df_user_features):,}")
    print(f"   Features per user: {len(df_user_features.columns)}")
    
    if len(df_user_features) > 0:
        print(f"   Avg sessions per user: {df_user_features['num_sessions'].mean():.2f}")
        print(f"   Avg turns per user: {df_user_features['num_turns'].mean():.2f}")
    else:
        print(f"   ⚠️ Warning: No users analyzed (check session_id mapping)")
    
    # Save checkpoint
    save_checkpoint(df_user_features, checkpoint_name,
                   "User-level features (aggregated across all sessions)")
    
    return df_user_features


# ============================================================================
# STEP 6: LOAD AND MERGE EVENT DATA
# ============================================================================

def load_and_merge_events(df_user_features):
    """Load event data and merge with user features"""
    checkpoint_name = "05_user_with_events"
    
    # Check checkpoint
    if checkpoint_exists(checkpoint_name):
        return load_checkpoint(checkpoint_name)
    
    # Check if we have any users to analyze
    if len(df_user_features) == 0:
        print("\n⚠️ WARNING: No users with valid user_id found!")
        print("   Cannot proceed with event data merging.")
        print("   Returning empty DataFrame.")
        return pd.DataFrame()
    
    print("\n" + "="*80)
    print("STEP 5: MERGING WITH EVENT DATA")
    print("="*80)
    
    print(f"\n📂 Loading: {DATA_FILES['event_bh'].name}")
    print("   (This may take a while for large files...)")
    
    # Load event data
    df_events = pd.read_csv(DATA_FILES['event_bh'])
    
    print(f"   Total events: {len(df_events):,}")
    print(f"   Unique users in events: {df_events['user_id'].nunique():,}")
    
    # Create user-level outcomes from events
    print("\n📊 Aggregating events by user...")
    
    user_events = df_events.groupby('user_id').agg({
        'event_name': ['count', lambda x: list(x)],
        'session_id': 'nunique',
        'begin_date': ['min', 'max']
    }).reset_index()
    
    user_events.columns = [
        'user_id', 'event_count', 'event_list', 
        'num_event_sessions', 'first_event_time', 'last_event_time'
    ]
    
    # Define success as having any of the key success events
    user_events['has_success_event'] = user_events['event_list'].apply(
        lambda events: any(e in SUCCESS_EVENTS for e in events)
    ).astype(int)
    
    # Count success events per user
    user_events['success_event_count'] = user_events['event_list'].apply(
        lambda events: sum(1 for e in events if e in SUCCESS_EVENTS)
    )
    
    # Drop event_list (too large for checkpoint)
    user_events = user_events.drop('event_list', axis=1)
    
    print(f"✅ User-level event aggregation:")
    print(f"   Unique users: {len(user_events):,}")
    print(f"   Users with success events: {user_events['has_success_event'].sum():,}")
    print(f"   Success rate: {user_events['has_success_event'].mean():.2%}")
    
    # Merge with user features
    print("\n🔗 Merging user features with event data...")
    df_regression = pd.merge(
        df_user_features,
        user_events[['user_id', 'event_count', 'has_success_event', 'success_event_count']],
        on='user_id',
        how='inner'
    )
    
    print(f"\n✅ Merge completed:")
    print(f"   Final dataset size: {len(df_regression):,} users")
    print(f"   Success users: {(df_regression['has_success_event'] == 1).sum():,}")
    print(f"   Failure users: {(df_regression['has_success_event'] == 0).sum():,}")
    print(f"   Success rate: {df_regression['has_success_event'].mean():.2%}")
    
    # Filter for multi-turn if configured
    if CONFIG['focus_multi_turn']:
        df_filtered = df_regression[df_regression['num_turns'] >= CONFIG['min_turns']].copy()
        print(f"\n   Filtering for multi-turn conversations (>={CONFIG['min_turns']} turns):")
        print(f"   Before: {len(df_regression):,} users")
        print(f"   After: {len(df_filtered):,} users")
        df_regression = df_filtered
    
    # Save checkpoint
    save_checkpoint(df_regression, checkpoint_name,
                   "User features merged with success events - ready for regression")
    
    return df_regression


# ============================================================================
# STEP 7: STATISTICAL ANALYSIS
# ============================================================================

def statistical_analysis(df_regression):
    """Perform statistical analysis comparing success vs failure"""
    print("\n" + "="*80)
    print("STEP 6: STATISTICAL ANALYSIS")
    print("="*80)
    
    # Check if we have any data
    if len(df_regression) == 0:
        print("\n⚠️ WARNING: No data available for statistical analysis!")
        return pd.DataFrame()
    
    success = df_regression[df_regression['has_success_event'] == 1]
    failure = df_regression[df_regression['has_success_event'] == 0]
    
    print(f"\nSample sizes:")
    print(f"  Success users: {len(success):,}")
    print(f"  Failure users: {len(failure):,}")
    
    # Compare empathy dimensions
    print(f"\n{'='*80}")
    print("EMPATHY DIMENSIONS COMPARISON")
    print("="*80)
    
    comparison_results = []
    for dim in ['cognitive_empathy', 'affective_empathy', 'empathy_concerns', 'total_empathy']:
        success_vals = success[dim]
        failure_vals = failure[dim]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(success_vals, failure_vals)
        
        # Effect size (Cohen's d)
        mean_diff = success_vals.mean() - failure_vals.mean()
        pooled_std = np.sqrt(((len(success)-1)*success_vals.std()**2 + 
                              (len(failure)-1)*failure_vals.std()**2) / 
                             (len(success) + len(failure) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        comparison_results.append({
            'Dimension': dim.replace('_', ' ').title(),
            'Success_Mean': success_vals.mean(),
            'Success_Std': success_vals.std(),
            'Failure_Mean': failure_vals.mean(),
            'Failure_Std': failure_vals.std(),
            'Mean_Diff': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.round(4).to_string())
    
    # Save results
    comparison_df.to_csv(OUTPUT_DIR / 'empathy_comparison_stats.csv', index=False)
    print(f"\n💾 Saved: empathy_comparison_stats.csv")
    
    return comparison_df


# ============================================================================
# STEP 8: LOGISTIC REGRESSION
# ============================================================================

def logistic_regression_analysis(df_regression):
    """Perform logistic regression analysis"""
    print("\n" + "="*80)
    print("STEP 7: LOGISTIC REGRESSION ANALYSIS")
    print("="*80)
    
    if len(df_regression) == 0:
        print("⚠️ WARNING: No data available for regression analysis!")
        return None, None
    
    if len(df_regression) < 30:
        print("⚠️ Insufficient data for regression (need at least 30 records)")
        return None, None
    
    # Prepare features and target
    feature_cols = ['cognitive_empathy', 'affective_empathy', 'empathy_concerns']
    control_cols = ['num_turns', 'avg_response_time']
    
    X_full = df_regression[feature_cols + control_cols].copy()
    y = df_regression['has_success_event'].copy()
    
    print(f"\nData Preparation:")
    print(f"  Sample size: {len(X_full):,}")
    print(f"  Features: {X_full.columns.tolist()}")
    print(f"  Target distribution:")
    print(f"    Success (1): {y.sum():,} ({y.mean():.1%})")
    print(f"    Failure (0): {(~y.astype(bool)).sum():,} ({(1-y.mean()):.1%})")
    
    # Handle missing values
    if X_full.isnull().any().any():
        print(f"\n  ⚠️ Filling missing values...")
        X_full = X_full.fillna(X_full.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"\n  Training set: {len(X_train):,}")
    print(f"  Test set: {len(X_test):,}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression
    print(f"\n{'='*80}")
    print("MODEL FITTING")
    print("="*80)
    
    log_model = LogisticRegression(random_state=CONFIG['random_state'], max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_test = log_model.predict(X_test_scaled)
    y_pred_proba_test = log_model.predict_proba(X_test_scaled)[:, 1]
    
    # Model evaluation
    print(f"\nModel Performance:")
    print(f"  Training accuracy: {log_model.score(X_train_scaled, y_train):.4f}")
    print(f"  Test accuracy: {log_model.score(X_test_scaled, y_test):.4f}")
    
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba_test)
        print(f"  AUC-ROC: {auc_score:.4f}")
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*80)
    print(classification_report(y_test, y_pred_test, target_names=['Failure', 'Success']))
    
    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE (COEFFICIENTS)")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'Feature': X_full.columns,
        'Coefficient': log_model.coef_[0],
        'Abs_Coefficient': np.abs(log_model.coef_[0]),
        'Odds_Ratio': np.exp(log_model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.round(4).to_string())
    
    print(f"\nInterpretation (Top 3 features):")
    for idx, row in feature_importance.head(3).iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        pct_change = (row['Odds_Ratio'] - 1) * 100
        print(f"  • {row['Feature']}: 1 SD increase {direction} odds of Success by {abs(pct_change):.1f}%")
    
    # Save results
    feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print(f"\n💾 Saved: feature_importance.csv")
    
    return log_model, feature_importance


# ============================================================================
# MARKDOWN REPORT GENERATION
# ============================================================================

def generate_markdown_report(df_regression, comparison_df, feature_importance, log_model):
    """Generate comprehensive analysis report in Markdown format"""
    
    report_path = OUTPUT_DIR / 'empathy_analysis_report.md'
    
    print("\n" + "="*80)
    print("GENERATING MARKDOWN REPORT")
    print("="*80)
    
    if len(df_regression) == 0:
        print("⚠️ No data available - skipping report generation")
        return
    
    success = df_regression[df_regression['has_success_event'] == 1]
    failure = df_regression[df_regression['has_success_event'] == 0]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Empathy Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"This report analyzes the relationship between empathy dimensions in chatbot conversations and user success events.\n\n")
        f.write(f"- **Total Users Analyzed:** {len(df_regression):,}\n")
        f.write(f"- **Success Users:** {len(success):,} ({len(success)/len(df_regression)*100:.1f}%)\n")
        f.write(f"- **Failure Users:** {len(failure):,} ({len(failure)/len(df_regression)*100:.1f}%)\n")
        f.write(f"- **Average Conversations per User:** {df_regression['num_sessions'].mean():.1f}\n")
        f.write(f"- **Average Turns per User:** {df_regression['num_turns'].mean():.1f}\n\n")
        
        # Dataset Overview
        f.write("---\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write("### User Distribution\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Users | {len(df_regression):,} |\n")
        f.write(f"| Users with Success Events | {len(success):,} |\n")
        f.write(f"| Users without Success Events | {len(failure):,} |\n")
        f.write(f"| Success Rate | {len(success)/len(df_regression)*100:.2f}% |\n\n")
        
        f.write("### Engagement Metrics\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        f.write(f"| Sessions per User | {df_regression['num_sessions'].mean():.2f} | {df_regression['num_sessions'].std():.2f} | {df_regression['num_sessions'].min():.0f} | {df_regression['num_sessions'].max():.0f} |\n")
        f.write(f"| Turns per User | {df_regression['num_turns'].mean():.2f} | {df_regression['num_turns'].std():.2f} | {df_regression['num_turns'].min():.0f} | {df_regression['num_turns'].max():.0f} |\n")
        if 'avg_response_time' in df_regression.columns:
            f.write(f"| Avg Response Time (s) | {df_regression['avg_response_time'].mean():.2f} | {df_regression['avg_response_time'].std():.2f} | {df_regression['avg_response_time'].min():.2f} | {df_regression['avg_response_time'].max():.2f} |\n")
        f.write("\n")
        
        # Empathy Analysis
        f.write("---\n\n")
        f.write("## 2. Empathy Dimensions Analysis\n\n")
        
        if comparison_df is not None and len(comparison_df) > 0:
            f.write("### Comparison between Success and Failure Groups\n\n")
            f.write("| Dimension | Success Mean (SD) | Failure Mean (SD) | Difference | t-statistic | p-value | Cohen's d | Significance |\n")
            f.write("|-----------|-------------------|-------------------|------------|-------------|---------|-----------|-------------|\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"| {row['Dimension']} | {row['Success_Mean']:.4f} ({row['Success_Std']:.4f}) | ")
                f.write(f"{row['Failure_Mean']:.4f} ({row['Failure_Std']:.4f}) | ")
                f.write(f"{row['Mean_Diff']:+.4f} | {row['t_statistic']:.3f} | ")
                f.write(f"{row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['Significant']} |\n")
            
            f.write("\n**Significance levels:** `***` p<0.001, `**` p<0.01, `*` p<0.05\n\n")
            
            # Key Findings
            f.write("### Key Findings\n\n")
            significant_dims = comparison_df[comparison_df['p_value'] < 0.05].sort_values('p_value')
            
            if len(significant_dims) > 0:
                f.write("**Statistically Significant Differences:**\n\n")
                for _, row in significant_dims.iterrows():
                    direction = "higher" if row['Mean_Diff'] > 0 else "lower"
                    effect_size = "large" if abs(row['cohens_d']) > 0.8 else "medium" if abs(row['cohens_d']) > 0.5 else "small"
                    f.write(f"- **{row['Dimension']}**: Success group shows {direction} levels ")
                    f.write(f"(difference = {abs(row['Mean_Diff']):.4f}, p = {row['p_value']:.4f}, ")
                    f.write(f"effect size = {effect_size}, Cohen's d = {row['cohens_d']:.3f})\n")
            else:
                f.write("No statistically significant differences found (p < 0.05).\n")
            f.write("\n")
        
        # Logistic Regression
        f.write("---\n\n")
        f.write("## 3. Predictive Modeling\n\n")
        
        if feature_importance is not None and len(feature_importance) > 0:
            f.write("### Logistic Regression Results\n\n")
            f.write("This model predicts the probability of success events based on empathy dimensions and control variables.\n\n")
            
            f.write("#### Feature Importance\n\n")
            f.write("| Feature | Coefficient | Odds Ratio | Impact |\n")
            f.write("|---------|-------------|------------|--------|\n")
            
            for _, row in feature_importance.iterrows():
                direction = "↑" if row['Coefficient'] > 0 else "↓"
                pct_change = abs((row['Odds_Ratio'] - 1) * 100)
                f.write(f"| {row['Feature']} | {row['Coefficient']:+.4f} | {row['Odds_Ratio']:.4f} | ")
                f.write(f"{direction} {pct_change:.1f}% |\n")
            
            f.write("\n**Interpretation:**\n\n")
            f.write("- **Positive Coefficient (↑):** Increases the odds of success\n")
            f.write("- **Negative Coefficient (↓):** Decreases the odds of success\n")
            f.write("- **Odds Ratio:** Multiplicative change in odds for 1 standard deviation increase\n\n")
            
            f.write("#### Top 3 Most Influential Features\n\n")
            top3 = feature_importance.head(3)
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                direction = "increases" if row['Coefficient'] > 0 else "decreases"
                pct_change = abs((row['Odds_Ratio'] - 1) * 100)
                f.write(f"{i}. **{row['Feature']}**: A 1 SD increase {direction} the odds of success by {pct_change:.1f}%\n")
            f.write("\n")
        
        # Empathy Score Distribution
        f.write("---\n\n")
        f.write("## 4. Empathy Score Distribution\n\n")
        
        empathy_cols = ['cognitive_empathy', 'affective_empathy', 'empathy_concerns', 'total_empathy']
        
        f.write("### Overall Distribution\n\n")
        f.write("| Dimension | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
        f.write("|-----------|------|-----|-----|-----|-----|-----|-----|\n")
        
        for col in empathy_cols:
            if col in df_regression.columns:
                stats_vals = df_regression[col].describe()
                f.write(f"| {col.replace('_', ' ').title()} | {stats_vals['mean']:.4f} | {stats_vals['std']:.4f} | ")
                f.write(f"{stats_vals['min']:.4f} | {stats_vals['25%']:.4f} | {stats_vals['50%']:.4f} | ")
                f.write(f"{stats_vals['75%']:.4f} | {stats_vals['max']:.4f} |\n")
        f.write("\n")
        
        # Recommendations
        f.write("---\n\n")
        f.write("## 5. Recommendations\n\n")
        
        if comparison_df is not None and len(comparison_df) > 0:
            significant_positive = comparison_df[(comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)]
            
            if len(significant_positive) > 0:
                f.write("Based on the analysis, the following empathy dimensions are positively associated with user success:\n\n")
                for _, row in significant_positive.iterrows():
                    f.write(f"- **{row['Dimension']}**: Focus on enhancing this dimension in chatbot responses\n")
                f.write("\n")
            
            f.write("### Actionable Insights\n\n")
            f.write("1. **Train chatbot models** to emphasize empathy dimensions that show positive correlation with success\n")
            f.write("2. **Monitor empathy metrics** in real-time conversations to identify at-risk users\n")
            f.write("3. **A/B test** different empathy strategies to optimize for user success\n")
            f.write("4. **Develop guidelines** for human agents based on successful empathy patterns\n\n")
        
        # Methodology
        f.write("---\n\n")
        f.write("## 6. Methodology\n\n")
        f.write("### Data Processing\n\n")
        f.write("1. **User-level aggregation**: All conversations and sessions aggregated by user\n")
        f.write("2. **Empathy scoring**: Three dimensions measured:\n")
        f.write("   - Cognitive Empathy: Understanding user's perspective\n")
        f.write("   - Affective Empathy: Emotional responsiveness\n")
        f.write("   - Empathy Concerns: Supportive behaviors\n")
        f.write("3. **Success definition**: Users with specific success events in event data\n\n")
        
        f.write("### Statistical Methods\n\n")
        f.write("- **Independent t-tests**: Compare means between success and failure groups\n")
        f.write("- **Cohen's d**: Measure effect sizes\n")
        f.write("- **Logistic regression**: Predict success probability from empathy dimensions\n")
        f.write("- **Odds ratios**: Quantify impact of each feature\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write("## Appendix\n\n")
        f.write("### Data Files\n\n")
        f.write(f"- **Main dataset**: `user_level_dataset.csv`\n")
        f.write(f"- **Statistical comparison**: `empathy_comparison_stats.csv`\n")
        f.write(f"- **Feature importance**: `feature_importance.csv`\n\n")
        
        f.write("### Contact\n\n")
        f.write("For questions about this analysis, please refer to the research team.\n\n")
        f.write("---\n\n")
        f.write("*End of Report*\n")
    
    print(f"✅ Markdown report saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    try:
        # Step 1: Load session_id -> user_id mapping from event data
        session_to_user = load_user_mapping()
        
        # Step 2: Load conversations and map user_id
        df_conversations = load_conversations(session_to_user)
        
        # Step 3: Merge turns
        df_turns = merge_consecutive_turns(df_conversations)
        
        # Step 4: Calculate user features
        df_user_features = analyze_users(df_turns)
        
        # Step 5: Merge with events
        df_regression = load_and_merge_events(df_user_features)
        
        # Step 6: Statistical analysis
        comparison_df = statistical_analysis(df_regression)
        
        # Step 7: Logistic regression
        log_model, feature_importance = logistic_regression_analysis(df_regression)
        
        # Step 8: Generate comprehensive Markdown report
        generate_markdown_report(df_regression, comparison_df, feature_importance, log_model)
        
        # Save final dataset
        print("\n" + "="*80)
        print("EXPORTING FINAL RESULTS")
        print("="*80)
        
        if len(df_regression) > 0:
            df_regression.to_csv(OUTPUT_DIR / 'user_level_dataset.csv', index=False)
            print(f"✅ Saved: user_level_dataset.csv ({len(df_regression):,} users)")
            
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE!")
            print("="*80)
            print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")
            print(f"Checkpoints saved to: {CHECKPOINT_DIR.resolve()}")
            print(f"\n📄 Main report: empathy_analysis_report.md")
            print(f"\nFinal dataset: {len(df_regression):,} users")
            print(f"Success rate: {df_regression['has_success_event'].mean():.2%}")
            print(f"\nAnalysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n⚠️ WARNING: No data to export!")
            print("   The analysis completed but no users were successfully mapped.")
            print("   Please check that session_ids in conversation files match event_bh.csv")
            print(f"\nAnalysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df_regression
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    df_results = main()

