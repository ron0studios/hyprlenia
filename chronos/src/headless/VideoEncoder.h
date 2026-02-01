#ifndef CHRONOS_VIDEO_ENCODER_H
#define CHRONOS_VIDEO_ENCODER_H

#include <cstdint>
#include <string>

// Forward declarations for FFmpeg types
struct AVFormatContext;
struct AVCodecContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct SwsContext;

class VideoEncoder {
 public:
  VideoEncoder();
  ~VideoEncoder();

  // Open video file for writing
  // Returns true on success
  bool open(const std::string& filename, int width, int height, int fps, int bitrate = 8000000);

  // Write a frame (RGB24 data, width*height*3 bytes)
  // Returns true on success
  bool writeFrame(const uint8_t* rgb24Data);

  // Finish encoding and close file
  void close();

  // Get frame count
  int64_t frameCount() const { return m_frameCount; }

 private:
  AVFormatContext* m_formatCtx;
  AVCodecContext* m_codecCtx;
  AVStream* m_stream;
  AVFrame* m_frame;
  AVPacket* m_packet;
  SwsContext* m_swsCtx;

  int m_width;
  int m_height;
  int m_fps;
  int64_t m_frameCount;
  bool m_initialized;

  bool encodeFrame(AVFrame* frame);
};

#endif  // CHRONOS_VIDEO_ENCODER_H
