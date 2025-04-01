#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>

// networking
// #include <sys/socket.h>
// #include <netinet/in.h>
// #include <arpa/inet.h>

// libav*
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#include "utils.h"
#include "streaming.h"


static AVCodec *codec = NULL;
static AVCodecContext *encoder = NULL;
static AVFormatContext *muxer = NULL;
static AVStream *video_track = NULL;
static AVFrame *frame = NULL, *yuv_frame = NULL;
static AVPacket *encoded_packet = NULL, *encoded_frame = NULL;
static struct SwsContext *sws_ctx = NULL;
static int64_t pts = 0;
static FILE *outfile = 0;

const int PORT = 8080;
const char *SERVER_ADDR = "127.0.0.1";
// static int sockfd = 0;
// static struct sockaddr_in addr;

void init_libav(const int width, const int height, const int count)
{
    outfile = fopen("out","wb");

    // assert((sockfd = socket(AF_INET, SOCK_STREAM, 0)) != -1);

    // memset(&addr,0,sizeof(struct sockaddr_in));
    // addr.sin_family = AF_INET;
    // addr.sin_addr.s_addr = inet_addr(SERVER_ADDR);
    // addr.sin_port = htons(PORT);
    // connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    // bind(sockfd, (struct sockaddr*)& addr, sizeof(addr));

#ifdef DEBUG
    av_log_set_level(AV_LOG_VERBOSE);
#else
    av_log_set_level(AV_LOG_ERROR);
#endif

    frame = av_frame_alloc();
    if(!frame){         fprintf(stderr, "Could not allocate video frame\n"); exit(1); }
    frame->format = AV_PIX_FMT_RGB24;
    frame->width  = width; 
    frame->height = height;

    int ret = av_frame_get_buffer(frame, count);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate the video frame data\n");
        exit(1);
    }
    ret = av_frame_make_writable(frame);
    if (ret < 0) exit(1);

    sws_ctx = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
    yuv_frame = av_frame_alloc();
    const int yuv_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, width, height, 1);
    yuv_frame->format = AV_PIX_FMT_YUV420P;
    yuv_frame->width  = width; 
    yuv_frame->height = height;
    ret = av_frame_get_buffer(yuv_frame, 16);
    assert(ret == 0);


    encoded_frame = av_packet_alloc();
    assert(encoded_frame);
    // encoded_packet = av_packet_alloc();

    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    // const char *codec_name = "rtp";  // TODO: change to use yuv422 pixel format as input
    // codec = avcodec_find_encoder_by_name(codec_name);

    if (!codec) {
        fprintf(stderr, "Error setting up codec\n");
        // fprintf(stderr, "Codec '%s' not found\n", codec_name);
        exit(1);
    }

    encoder = avcodec_alloc_context3(codec);
    assert(encoder);
    encoder->bit_rate = 10 * 1024 * 1024;
    encoder->width = width;
    encoder->height = height;
    encoder->time_base = (AVRational) {1,30};
    encoder->framerate = (AVRational){15, 1};
    encoder->gop_size = 15;
    encoder->max_b_frames = 1;
    encoder->pix_fmt = AV_PIX_FMT_YUV420P;
    av_opt_set(encoder->priv_data, "preset", "fast", 0);
    ret = avcodec_open2(encoder, codec, NULL);
    assert(ret == 0);

    // set up muxer
    muxer = avformat_alloc_context();
    // avformat_alloc_output_context2(&muxer, NULL, "flv", RTMP_URL)
    muxer->oformat = av_guess_format("matroska", "test.mkv", NULL);

    video_track = avformat_new_stream(muxer, NULL);
    // muxer->oformat->video_codec = AV_CODEC_ID_H264;
    // AVStream* audio_track = avformat_new_stream(muxer, NULL);
    // muxer->oformat->audio_codec = AV_CODEC_ID_OPUS;

    avcodec_parameters_from_context(video_track->codecpar, encoder); 
    video_track->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    video_track->time_base = (AVRational) {1,30};
    video_track->avg_frame_rate = (AVRational) {30, 1};

    return;
}

static int stream_frame(AVFrame* frame){
    if(frame != NULL) frame->pts = ++pts;
    int ret = avcodec_send_frame(encoder, frame);
    if(ret != 0) av_errno_exit("AVCodec_send_frame", ret);

    do{
        ret = avcodec_receive_packet(encoder, encoded_frame);
        if(ret == 0) {
            encoded_frame->pts = ++pts;
            fprintf(stderr,"Streaming frame...\n");

            // send(sockfd, encoded_frame->data, encoded_frame->size, MSG_NOSIGNAL);
            // fprintf(stderr, "Packet size: %d\n",encoded_frame->size);
            // int send_ret = sendto(sockfd, encoded_frame->data, encoded_frame->size, 0, (struct sockaddr*)&addr, sizeof(addr));
            // if(send_ret == -1){
                // fprintf(stderr,"Sendto: %s\n", strerror(errno));
                // exit(-1);
            // }
            fwrite(encoded_frame->data, encoded_frame->size, 1, outfile);

            AVRational encoder_time_base = (AVRational) {1, 30};
            encoded_frame->stream_index = video_track->index;
            int64_t scaled_dts = av_rescale_q(encoded_frame->dts, encoder_time_base, video_track->time_base);
            // input.packet.dts = scaled_dts;

            fprintf(stderr,"...\n");
            ret = av_write_frame(muxer, encoded_frame);

            if(ret == -1){
                av_errno_exit("AV_write_frame",ret);
            }

        }else if(ret == AVERROR_EOF){ break;
        }else if(ret == AVERROR(EAGAIN)){ continue;
        }else{
            av_errno_exit("AVCodec_send_frame",ret);
        }
        av_packet_unref(encoded_frame); 
    }while(frame == NULL);
    return 0;
}

void uninit_libav()
{
    stream_frame(NULL);
    fclose(outfile);

    avformat_free_context(muxer);
    // avio_format_context_unref(video_track);
    // av_packet_free(&encoded_packet); 
    if(encoded_packet != NULL) av_packet_unref(encoded_packet); 


    avcodec_free_context(&encoder);
    sws_freeContext(sws_ctx);
    // av_frame_free(&yuv_frame);
    av_frame_unref(yuv_frame);
    // av_frame_free(&frame);
    av_frame_unref(frame);
    // close(sockfd);
}

int send_frame(void *data, const int source_width, const int source_height)
{
    // fprintf(stderr,"%dx%d\n",source_width,source_height);
    // fprintf(stderr,"linesize = %d\n",frame->linesize[pts]);

    int ret = av_frame_make_writable(frame);
    if (ret < 0) exit(1);

    memcpy(frame->data[0],data,source_width*source_height*3);
    
    ret = av_frame_make_writable(yuv_frame);
    if (ret < 0) exit(1);

    // Convert the RGB frame to YUV420p
    sws_scale(sws_ctx, (const uint8_t *const *)frame->data, frame->linesize, 0, source_height, yuv_frame->data, yuv_frame->linesize);

    // fprintf(stderr,"!%d || !%d\n",avcodec_is_open(encoder), av_codec_is_encoder(encoder->codec));

    stream_frame(yuv_frame);

    return 0;
}
