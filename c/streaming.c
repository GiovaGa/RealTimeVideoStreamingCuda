#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static AVCodec *codec = NULL;
static AVCodecContext *encoder = NULL;
static AVFrame *frame = NULL;
static AVPacket *encoded_packet = NULL, *encoded_frame = NULL;
static int pts = 0;

void init_libav(const int width, const int height, const int count)
{
    av_log_set_level(AV_LOG_ERROR);
    av_log_set_level(AV_LOG_VERBOSE);

    frame = av_frame_alloc();
    if(!frame){         fprintf(stderr, "Could not allocate video frame\n"); exit(1); }
    frame->format = AV_PIX_FMT_YUV422P;
    frame->width  = width; 
    frame->height = height;

    int ret = av_frame_get_buffer(frame, count);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate the video frame data\n");
        exit(1);
    }
    ret = av_frame_make_writable(frame);
    if (ret < 0) exit(1);

    encoded_frame = av_packet_alloc();

    codec = avcodec_find_encoder(AV_CODEC_ID_H264);

    encoder = avcodec_alloc_context3(codec);
    assert(encoder);
    encoder->bit_rate = 10 * 1000 * 10000;
    encoder->width = width;
    encoder->height = height;
    encoder->time_base = (AVRational) {1,30};
    encoder->gop_size = 15;
    encoder->max_b_frames = 1;
    encoder->pix_fmt = AV_PIX_FMT_YUV420P;
    av_opt_set(encoder->priv_data, "preset", "fast", 0);
    ret = avcodec_open2(encoder, codec, NULL);
    assert(ret == 0);

    // set up muxer
    AVFormatContext* muxer = avformat_alloc_context();
    // avformat_alloc_output_context2(&muxer, NULL, "flv", RTMP_URL)
    muxer->oformat = av_guess_format("matroska", "test.mkv", NULL);

    AVStream* video_track = avformat_new_stream(muxer, NULL);
    // muxer->oformat->video_codec = AV_CODEC_ID_H264;
    // AVStream* audio_track = avformat_new_stream(muxer, NULL);
    // muxer->oformat->audio_codec = AV_CODEC_ID_OPUS;

    avcodec_parameters_from_context(video_track->codecpar, encoder); 
    video_track->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

    video_track->time_base = (AVRational) {1,30};
    video_track->avg_frame_rate = (AVRational) {30, 1};

    //int avio_open2 (  AVIOContext **   s, const char *   url, int   flags, const AVIOInterruptCB *   int_cb, AVDictionary **   options )
}

void uninit_libav()
{
    // av_packet_free(&encoded_packet); 
    av_packet_unref(encoded_frame); 
    if(encoded_packet != NULL) av_packet_unref(encoded_packet); 
    // av_frame_free(&frame);
    av_frame_unref(frame);
}


int send_frame(void *data, const int source_width, const int source_height)
{
    fprintf(stderr,"%dx%d\n",source_width,source_height);
    // fprintf(stderr,"linesize = %d\n",frame->linesize[pts]);

    int ret = av_frame_make_writable(frame);
    if (ret < 0) exit(1);

    memcpy(frame->data[pts],data,source_width*source_height*2);
    // frame->pts = pts++;
    // fprintf(stderr,"%d \t%d\t%d\t%d\n", data, frame->data[0]);

    // for(int y = 0;y < frame->height;++y){
        // for(int x = 0;x < frame->width;++x){
            // for(int ch = 0;ch < 3;++ch)
                // frame->data[0][(y*frame->linesize[ch]) + 3*x+ch] = ((uint8_t*)data)[3*(y*source_width + x) + ch];
            // }
        // }
    // Convert the frame from RGB to YUV420p
    struct SwsContext *sws_ctx = sws_getContext(source_width, source_height, AV_PIX_FMT_RGB24, source_width, source_height, AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
    AVFrame *yuv_frame = av_frame_alloc();
    const int yuv_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, source_width, source_height, 1);
    uint8_t *yuv_data = (uint8_t *)av_malloc(yuv_size);
    av_image_fill_arrays(yuv_frame->data, yuv_frame->linesize, yuv_data, AV_PIX_FMT_YUV420P, source_width, source_height, 1);

    // fprintf(stderr,"%d\n",frame->linesize);
    // Convert the RGB frame to YUV420p
    sws_scale(sws_ctx, (const uint8_t *const *)frame->data, frame->linesize, 0, source_height, yuv_frame->data, yuv_frame->linesize);
    assert(ret == 0);

    ret = avcodec_send_frame(encoder, yuv_frame);
    fprintf(stderr,"%s\n",av_err2str(ret));
    assert(ret == 0);
    ret = avcodec_receive_packet(encoder, encoded_frame);
    fprintf(stderr,"Ok: line %d\n",__LINE__);
    if(ret == 0) {
        fprintf(stderr,"Encoding went well!\n");
    }else{
        fprintf(stderr,"Encoding didn't go well!\n");
        fprintf(stderr,"%s\n",av_err2str(ret));
    }

    FILE *outfile = fopen("out","wb");

    fwrite(encoded_frame->data, 1, encoded_frame->size, outfile);

    fclose(outfile);
    sws_freeContext(sws_ctx);
    av_freep(&yuv_data);


    return 0;
    /* AVRational encoder_time_base = (AVRational) {1, 60};
    encoded_packet.stream_index = video_track->index;

    int64_t scaled_pts = av_rescale_q(encoded_packet.pts, encoder_time_base, video_track->time_base);
    encoded_packet.pts = scaled_pts;

    int64_t scaled_dts = av_rescale_q(encoded_packet.dts, encoder_time_base, video_track->time_base);
    input.packet.dts = scaled_dts;

    ret = av_write_frame(muxer->av_format_context, &encoded_packet); */
}
