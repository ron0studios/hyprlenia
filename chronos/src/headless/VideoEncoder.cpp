#include "headless/VideoEncoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <iostream>

VideoEncoder::VideoEncoder()
    : m_formatCtx(nullptr),
      m_codecCtx(nullptr),
      m_stream(nullptr),
      m_frame(nullptr),
      m_packet(nullptr),
      m_swsCtx(nullptr),
      m_width(0),
      m_height(0),
      m_fps(30),
      m_frameCount(0),
      m_initialized(false) {}

VideoEncoder::~VideoEncoder() {
  close();
}

bool VideoEncoder::open(const std::string& filename, int width, int height, int fps, int bitrate) {
  m_width = width;
  m_height = height;
  m_fps = fps;

  // Allocate output format context
  int ret = avformat_alloc_output_context2(&m_formatCtx, nullptr, nullptr, filename.c_str());
  if (ret < 0 || !m_formatCtx) {
    std::cerr << "ERROR: Could not create output context for " << filename << std::endl;
    return false;
  }

  // Find H.264 encoder
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (!codec) {
    std::cerr << "ERROR: H.264 codec not found" << std::endl;
    close();
    return false;
  }

  // Create stream
  m_stream = avformat_new_stream(m_formatCtx, nullptr);
  if (!m_stream) {
    std::cerr << "ERROR: Could not create video stream" << std::endl;
    close();
    return false;
  }

  // Allocate codec context
  m_codecCtx = avcodec_alloc_context3(codec);
  if (!m_codecCtx) {
    std::cerr << "ERROR: Could not allocate codec context" << std::endl;
    close();
    return false;
  }

  // Set codec parameters
  m_codecCtx->width = width;
  m_codecCtx->height = height;
  m_codecCtx->time_base = AVRational{1, fps};
  m_codecCtx->framerate = AVRational{fps, 1};
  m_codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
  m_codecCtx->bit_rate = bitrate;
  m_codecCtx->gop_size = fps;  // Keyframe every second
  m_codecCtx->max_b_frames = 2;

  // Set H.264 preset for quality/speed tradeoff
  av_opt_set(m_codecCtx->priv_data, "preset", "medium", 0);
  av_opt_set(m_codecCtx->priv_data, "crf", "23", 0);  // Constant rate factor

  // Some formats want stream headers separate
  if (m_formatCtx->oformat->flags & AVFMT_GLOBALHEADER) {
    m_codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // Open codec
  ret = avcodec_open2(m_codecCtx, codec, nullptr);
  if (ret < 0) {
    std::cerr << "ERROR: Could not open codec" << std::endl;
    close();
    return false;
  }

  // Copy codec parameters to stream
  ret = avcodec_parameters_from_context(m_stream->codecpar, m_codecCtx);
  if (ret < 0) {
    std::cerr << "ERROR: Could not copy codec parameters" << std::endl;
    close();
    return false;
  }

  m_stream->time_base = m_codecCtx->time_base;

  // Allocate frame
  m_frame = av_frame_alloc();
  if (!m_frame) {
    std::cerr << "ERROR: Could not allocate frame" << std::endl;
    close();
    return false;
  }

  m_frame->format = m_codecCtx->pix_fmt;
  m_frame->width = width;
  m_frame->height = height;

  ret = av_frame_get_buffer(m_frame, 0);
  if (ret < 0) {
    std::cerr << "ERROR: Could not allocate frame buffer" << std::endl;
    close();
    return false;
  }

  // Allocate packet
  m_packet = av_packet_alloc();
  if (!m_packet) {
    std::cerr << "ERROR: Could not allocate packet" << std::endl;
    close();
    return false;
  }

  // Create swscale context for RGB24 -> YUV420P conversion
  m_swsCtx = sws_getContext(
      width, height, AV_PIX_FMT_RGB24,
      width, height, AV_PIX_FMT_YUV420P,
      SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!m_swsCtx) {
    std::cerr << "ERROR: Could not create swscale context" << std::endl;
    close();
    return false;
  }

  // Open output file
  if (!(m_formatCtx->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&m_formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      std::cerr << "ERROR: Could not open output file: " << filename << std::endl;
      close();
      return false;
    }
  }

  // Write file header
  ret = avformat_write_header(m_formatCtx, nullptr);
  if (ret < 0) {
    std::cerr << "ERROR: Could not write file header" << std::endl;
    close();
    return false;
  }

  m_initialized = true;
  std::cout << "VideoEncoder opened: " << filename << " (" << width << "x" << height << " @ " << fps << " fps)" << std::endl;
  return true;
}

bool VideoEncoder::writeFrame(const uint8_t* rgb24Data) {
  if (!m_initialized) return false;

  // Make frame writable
  int ret = av_frame_make_writable(m_frame);
  if (ret < 0) {
    std::cerr << "ERROR: Could not make frame writable" << std::endl;
    return false;
  }

  // Convert RGB24 to YUV420P
  const uint8_t* srcSlice[1] = {rgb24Data};
  int srcStride[1] = {m_width * 3};

  sws_scale(m_swsCtx, srcSlice, srcStride, 0, m_height,
            m_frame->data, m_frame->linesize);

  // Set presentation timestamp
  m_frame->pts = m_frameCount++;

  return encodeFrame(m_frame);
}

bool VideoEncoder::encodeFrame(AVFrame* frame) {
  // Send frame to encoder
  int ret = avcodec_send_frame(m_codecCtx, frame);
  if (ret < 0) {
    std::cerr << "ERROR: Error sending frame to encoder" << std::endl;
    return false;
  }

  // Receive encoded packets
  while (ret >= 0) {
    ret = avcodec_receive_packet(m_codecCtx, m_packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return true;  // Need more frames or done
    }
    if (ret < 0) {
      std::cerr << "ERROR: Error encoding frame" << std::endl;
      return false;
    }

    // Rescale timestamps
    av_packet_rescale_ts(m_packet, m_codecCtx->time_base, m_stream->time_base);
    m_packet->stream_index = m_stream->index;

    // Write packet
    ret = av_interleaved_write_frame(m_formatCtx, m_packet);
    if (ret < 0) {
      std::cerr << "ERROR: Error writing packet" << std::endl;
      return false;
    }
  }

  return true;
}

void VideoEncoder::close() {
  if (m_initialized) {
    // Flush encoder
    encodeFrame(nullptr);

    // Write trailer
    av_write_trailer(m_formatCtx);

    std::cout << "VideoEncoder closed: " << m_frameCount << " frames written" << std::endl;
  }

  if (m_swsCtx) {
    sws_freeContext(m_swsCtx);
    m_swsCtx = nullptr;
  }

  if (m_packet) {
    av_packet_free(&m_packet);
    m_packet = nullptr;
  }

  if (m_frame) {
    av_frame_free(&m_frame);
    m_frame = nullptr;
  }

  if (m_codecCtx) {
    avcodec_free_context(&m_codecCtx);
    m_codecCtx = nullptr;
  }

  if (m_formatCtx) {
    if (!(m_formatCtx->oformat->flags & AVFMT_NOFILE) && m_formatCtx->pb) {
      avio_closep(&m_formatCtx->pb);
    }
    avformat_free_context(m_formatCtx);
    m_formatCtx = nullptr;
  }

  m_stream = nullptr;
  m_initialized = false;
  m_frameCount = 0;
}
