import os
from googleapiclient.discovery import build
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up YouTube API key and build the API service
API_KEY = 'AIzaSyAn0D-UjVomkTZqXm_Klh4Ui3SJm84eiH0'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Sentiment Analysis Setup
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Route to fetch data by Channel ID
@app.route('/youtube/<channel_id>', methods=['POST'])
def fetch_channel_data(channel_id):
    try:
        # Fetch channel details
        channel_response = youtube.channels().list(
            id=channel_id,
            part='snippet,statistics'
        ).execute()

        print("Channel Response:", channel_response)  # Debug log

        if 'items' not in channel_response or not channel_response['items']:
            return jsonify({'error': 'No channel found with the provided ID.'}), 404

        channel_stats = [{
            'Channel_name': channel_response['items'][0]['snippet']['title'],
            'Subscribers': int(channel_response['items'][0]['statistics'].get('subscriberCount', 0)),
            'Views': int(channel_response['items'][0]['statistics'].get('viewCount', 0)),
            'Total_videos': int(channel_response['items'][0]['statistics'].get('videoCount', 0)),
        }]

        # Fetch video details for the channel
        video_response = youtube.search().list(
            channelId=channel_id,
            part='snippet',
            order='date',
            maxResults=10
        ).execute()

        video_ids = []
        for item in video_response.get('items', []):
            if 'videoId' in item['id']:
                video_ids.append(item['id']['videoId'])

        if not video_ids:
            return jsonify({'error': 'No videos found for this channel.'}), 404

        # Fetch statistics for each video
        videos = youtube.videos().list(
            id=','.join(video_ids),
            part='snippet,statistics'
        ).execute()

        video_details = []
        for video in videos['items']:
            snippet = video['snippet']
            stats = video['statistics']

            video_data = {
                'Title': snippet['title'],
                'Published_date': snippet['publishedAt'],
                'Views': int(stats.get('viewCount', 0)),
                'Likes': int(stats.get('likeCount', 0)),
                'Comments': int(stats.get('commentCount', 0)),
            }
            video_details.append(video_data)

        return jsonify({'channel_stats': channel_stats, 'video_details': video_details})

    except Exception as e:
        print(f"Error fetching channel data: {e}")
        return jsonify({'error': str(e)}), 500


# Route to fetch top videos by search term
@app.route('/youtube/search/<search_term>', methods=['POST'])
def search_top_videos(search_term):
    try:
        # Fetch videos related to the search term
        search_response = youtube.search().list(
            q=search_term,
            part='snippet',
            type='video',
            maxResults=10
        ).execute()

        # Retrieve video IDs and thumbnails
        top_videos = []
        for item in search_response['items']:
            video_data = {
                'Title': item['snippet']['title'],
                'Published_date': item['snippet']['publishedAt'],
                'Thumbnail': item['snippet']['thumbnails']['high']['url'],
                'videoId': item['id']['videoId'],
            }
            top_videos.append(video_data)

        return jsonify({'top_videos': top_videos})

    except Exception as e:
        print(f"Error fetching video data: {e}")
        return jsonify({'error': 'Error fetching video data'}), 500


# Route to analyze comments of a video
@app.route('/youtube/comments/<video_id>', methods=['POST'])
def analyze_video_comments(video_id):
    try:
        print("Received Video ID:", video_id)  # Debug log
        comments = fetch_all_comments(video_id)
        total_comments = len(comments)

        if total_comments == 0:
            return jsonify({
                'total_comments': total_comments,
                'positive_comments': 0,
                'negative_comments': 0
            })

        # Analyze comments
        analysis_results = analyze_comments(comments)

        positive_comments = sum(1 for result in analysis_results if result['sentiment_category'] == 'Positive')
        negative_comments = sum(1 for result in analysis_results if result['sentiment_category'] == 'Negative')

        return jsonify({
            'total_comments': total_comments,
            'positive_comments': positive_comments,
            'negative_comments': negative_comments
        })

    except Exception as e:
        print(f"Error analyzing comments: {e}")
        return jsonify({'error': str(e)}), 500


def fetch_all_comments(video_id):
    comments = []
    try:
        # Initial API request
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100  # Maximum allowed per request
        ).execute()
        
        # Continue fetching comments until there are no more pages
        while response:
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            
            # Check if there's another page of comments
            if 'nextPageToken' in response:
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=100,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                # No more comments to fetch
                break

        print(f"Fetched a total of {len(comments)} comments.")
    except Exception as e:
        print(f"Error fetching comments: {e}")
    
    return comments


def analyze_comments(comments):
    results = []
    for comment in comments:
        sentiment = sentiment_pipeline(comment)[0]
        sentiment_label = sentiment['label']
        sentiment_score = sentiment['score']
        
        # Map sentiment to stars and classify sentiment
        if sentiment_label == '0 stars':
            stars = 0
            sentiment_category = 'Negative'
        elif sentiment_label == '1 star':
            stars = 1
            sentiment_category = 'Negative'
        elif sentiment_label == '2 stars':
            stars = 2
            sentiment_category = 'Neutral'
        elif sentiment_label == '3 stars':
            stars = 3
            sentiment_category = 'Positive'
        elif sentiment_label == '4 stars':
            stars = 4
            sentiment_category = 'Positive'
        else:
            stars = 5
            sentiment_category = 'Positive'
        
        results.append({
            'comment': comment,
            'sentiment': sentiment_label,
            'score': sentiment_score,
            'stars': stars,
            'sentiment_category': sentiment_category
        })
    
    return results


if __name__ == '__main__':
    app.run(debug=True)
